from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from itertools import islice # Could be something else
from matplotlib.colors import CenteredNorm
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


attribution_palette = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)


def batch_with_cycle(iterable, n):
    values = tuple(iterable)
    iterator = iter(iterable)
    i = 0
    while batch := tuple(islice(iterator, n)):
        if len(batch) != n:
            batch = (values[i], *batch)
        yield batch
        i += 1


class OptimalTreePathFinder:
    """
    Find the optimal path to a leaf node in a sklearn decision tree.
    For classifiers: finds the leaf with maximum positive class proportion.
    For regressors: finds the leaf with maximum predicted value.
    """
    
    def __init__(self, tree):
        self.tree = tree
        self.children_left = tree.tree_.children_left
        self.children_right = tree.tree_.children_right
        self.feature = tree.tree_.feature
        self.threshold = tree.tree_.threshold
        self.value = tree.tree_.value
        
    def _get_leaf_score(self, node_id):
        """Get score for a leaf node."""
        if isinstance(self.tree, DecisionTreeClassifier):
            # Proportion of positive class (class 1)
            class_counts = self.value[node_id, 0]
            return class_counts[1] / class_counts.sum() if class_counts.sum() > 0 else 0
        else:  # Regressor
            return self.value[node_id, 0, 0]
    
    def find_best_path(self):
        """
        Traverse all paths from root to leaves and return the path to the 
        leaf with the best score.
        
        Returns:
            tuple: (path_list, leaf_node_id, best_score)
        """
        self.best_leaf = None
        self.best_score = -np.inf
        self.best_path = []
        
        self._traverse(0, [])
        
        return self.best_path, self.best_leaf, self.best_score
    
    def _traverse(self, node_id, current_path):
        """Recursively traverse tree and track best leaf."""
        child_left = self.children_left[node_id]
        child_right = self.children_right[node_id]
        
        # Leaf node: check if best so far
        if child_left == child_right:
            leaf_score = self._get_leaf_score(node_id)
            if leaf_score > self.best_score:
                self.best_score = leaf_score
                self.best_leaf = node_id
                # We only want split nodes in the max path
                self.best_path = current_path + [node_id]
        else:
            # Internal node: recurse on both children
            self._traverse(child_left, current_path + [node_id])
            self._traverse(child_right, current_path + [node_id])


def tree_to_gating_max_path(
        tree,
        X,
        axes,
        norm=None,
        hue=None,
        feature_names_in=None,
        plot_rectangles=True,
        dirty_display_trick=False
    ):
    # Find the optimal path to the best leaf
    path_finder = OptimalTreePathFinder(tree)
    max_path, leaf_node, best_score = path_finder.find_best_path()

    children_left = tree.tree_.children_left
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    if isinstance(tree, DecisionTreeRegressor):
        value = tree.tree_.value
    elif isinstance(tree, DecisionTreeClassifier):
        value = tree.tree_.value[:, 0, 1]
    else:
        raise NotImplementedError()

    
    # Iterate on split nodes
    iterator = tuple(batch_with_cycle(max_path[:-1], 2))
    if hue is None:
        hue = tree.predict(X)

    for i, (node_x, node_y) in enumerate(iterator):
        feature_x = feature[node_x]
        feature_y = feature[node_y]

        if feature_names_in is None:
            feature_names_in = tree.feature_names_in_

        print(feature_names_in[feature_x])
        print(feature_names_in[feature_y])
        x_name = feature_names_in[feature_x]
        y_name = feature_names_in[feature_y]
        
        if dirty_display_trick:
            # Quick trick to always have CD33 on the x-axis
            if y_name == "CD33 PC5.5" or x_name=="SS-A":
                temp = y_name
                y_name = x_name
                x_name = temp
        

        sns.scatterplot(
            data=X,
            x=x_name,
            y=y_name,
            s=10,
            hue=hue,
            hue_norm=norm,
            palette=attribution_palette,
            legend=False,
            ax=axes[i],
        )

        if plot_rectangles:
            xmin = X[x_name].min()
            xmax = X[x_name].max()
            ymin = X[y_name].min()
            ymax = X[y_name].max()
            x_thresh = threshold[node_x]
            y_thresh = threshold[node_y]

            x_thresh = threshold[node_x]
            rect_kwargs = dict(ls="--", ec="tab:red", fc="none")
            next_node_idx = 2 + 2*i

            # x goes to left
            if children_left[node_x] == node_y:
                # y goes to left
                if children_left[node_y] == max_path[next_node_idx]:
                    rect = Rectangle(
                            (xmin, ymin),
                            height=y_thresh- ymin,
                            width=x_thresh - xmin,
                            **rect_kwargs
                        )
                # y goes to right
                else:
                    rect = Rectangle(
                            (xmin, y_thresh),
                            height=ymax - y_thresh,
                            width=x_thresh - xmin,
                            **rect_kwargs
                        )

            # x goes to right
            else:
                # y goes to left
                if children_left[node_y] == max_path[next_node_idx]:
                    rect = Rectangle(
                            (x_thresh, ymin),
                            height=y_thresh - ymin,
                            width=xmax - x_thresh,
                            **rect_kwargs
                        )
                else:
                    # y goes to right
                    rect = Rectangle(
                            (x_thresh, y_thresh),
                            height=ymax - y_thresh,
                            width=xmax - x_thresh,
                            **rect_kwargs
                        )

            axes[i].add_patch(rect)

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return axes


def tree_to_gating_rectangles(tree, X):

    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    features_iterator = batch_with_cycle(np.unique(feature[feature > 0]), 2)
    features_iterator = tuple(features_iterator)
    n_biplots = len(features_iterator)

    fig, axes = plt.subplots(
        n_biplots,
        1,
        figsize=(5, 4 * n_biplots)
    )

    left_color = "orange"
    alpha=0.2

    for i, pair in enumerate(features_iterator):
        x_nodes = np.where(feature == pair[0])[0]
        y_nodes = np.where(feature == pair[1])[0]
        x_thresh = threshold[x_nodes]
        y_thresh = threshold[y_nodes]
        x_name = tree.feature_names_in_[pair[0]]
        y_name = tree.feature_names_in_[pair[1]]

        line_kwargs = dict(colors="black", linestyle="dashed")
        xmin = X[x_name].min()
        xmax = X[x_name].max()
        ymin = X[y_name].min()
        ymax = X[y_name].max()
        axes[i].hlines(y_thresh, xmin=xmin, xmax=xmax, **line_kwargs)
        axes[i].vlines(x_thresh, ymin=ymin, ymax=ymax, **line_kwargs)
        for x_node in x_nodes:
            x_thresh = threshold[x_node]
            if value[children_left[x_node]] > value[children_right[x_node]]:
                rect = Rectangle(
                    (xmin, ymin),
                    height=ymax - ymin,
                    width=x_thresh - xmin,
                    color=left_color,
                    alpha=alpha
                )
            else:
                rect = Rectangle(
                    (x_thresh, ymin),
                    height=ymax - ymin,
                    width=xmax - x_thresh,
                    color=left_color,
                    alpha=alpha
                )
            axes[i].add_patch(rect)

        for y_node in y_nodes:
            y_thresh = threshold[y_node]
            if value[children_left[y_node]] > value[children_right[y_node]]:
                rect = Rectangle(
                    (xmin, ymin),
                    height=y_thresh - ymin,
                    width=xmax - xmin,
                    color=left_color,
                    alpha=alpha
                )
            else:
                rect = Rectangle(
                    (xmin, y_thresh),
                    height=ymax - y_thresh,
                    width=xmax - xmin,
                    color=left_color,
                    alpha=alpha
                )
            axes[i].add_patch(rect)
        sns.scatterplot(
            data=X,
            x=x_name,
            y=y_name,
            s=10,
            color="gray",
            #palette=attribution_palette,
            legend=False,
            ax=axes[i]
        )
    return fig

def tree_to_gating(tree, X, y=None, add_cbar=False): 
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    features_iterator = batch_with_cycle(np.unique(feature[feature > 0]), 2)
    features_iterator = tuple(features_iterator)
    n_biplots = len(features_iterator)

    fig, axes = plt.subplots(
        n_biplots,
        1,
        figsize=(5, 4 * n_biplots)
    )

    # Create a single ScalarMappable for the color bar
    norm = CenteredNorm(vcenter=0)
    sm = plt.cm.ScalarMappable(cmap=attribution_palette, norm=norm)
    sm.set_array([])

    for i, pair in enumerate(features_iterator):
        x_nodes = np.where(feature == pair[0])
        y_nodes = np.where(feature == pair[1])
        x_thresh = threshold[x_nodes]
        y_thresh = threshold[y_nodes]
        x_name = tree.feature_names_in_[pair[0]]
        y_name = tree.feature_names_in_[pair[1]]
        if y is None:
            hue = tree.predict(X)
        else:
            hue = y

        sns.scatterplot(
            data=X,
            x=x_name,
            y=y_name,
            s=10,
            hue=hue,
            hue_norm=norm,
            palette=attribution_palette,
            legend=False,
            ax=axes[i]
        )

        line_kwargs = dict(colors="black", linestyle="dashed")
        axes[i].hlines(y_thresh, xmin=X[x_name].min(), xmax=X[x_name].max(), **line_kwargs)
        axes[i].vlines(x_thresh, ymin=X[y_name].min(), ymax=X[y_name].max(), **line_kwargs)

    # Add a single color bar for all subplots
    if add_cbar:
        fig.subplots_adjust(right=0.85)  # Adjust right margin to fit the color bar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Predicted $\\Delta_y$', size=15, loc="center", labelpad=10)
    return fig
    

