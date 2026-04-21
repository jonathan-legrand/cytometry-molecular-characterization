import flowkit as fk
from pathlib import Path

import numpy as np

def get_clipped_idx(col):
    max_val = col.max()
    min_val = col.max()
    msk = np.logical_or(np.isclose(col, max_val), np.isclose(col, min_val))
    return msk

logicle_pars = [275000,0.75,4.5,0.]
logicle_xform = fk.transforms.LogicleTransform(
    param_t=logicle_pars[0],
    param_w=logicle_pars[1],
    param_m=logicle_pars[2],
    param_a=logicle_pars[3]
)

class PreprocSample(fk.Sample):
    def __init__(self, fpath, *args, **kwargs):

        super().__init__(fpath, *args, **kwargs)

        # Correct channel idx issues
        if self.pnn_labels[0]=='FS PEAK':
            self.fluoro_indices=[3,4,5,6,7,8,9,10,11,12]
            self.scatter_indices=[1,2]
        else:
            self.fluoro_indices=[2,3,4,5,6,7,8,9,10,11]
            self.scatter_indices=[0,1]

        try:
            self.tube = fpath.name.split("_")[1].split(" ")[-1]
            self.patient = fpath.name.split("_")[0]
        except IndexError as e:
            print(self.original_filename)
            raise e
    
    @property
    def raw_shape(self):
        return self._raw_events.shape
        
        


class CustomSample(fk.Sample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Correct channel idx issues
        if self.pnn_labels[0]=='FS PEAK':
            self.fluoro_indices=[3,4,5,6,7,8,9,10,11,12]
            self.scatter_indices=[1,2]
        else:
            self.fluoro_indices=[2,3,4,5,6,7,8,9,10,11]
            self.scatter_indices=[0,1]

        spill = self.get_metadata()['spill']

        detectors = [self.pnn_labels[i] for i in self.fluoro_indices]
        fluorochromes = [self.pns_labels[i] for i in self.fluoro_indices]

        self._remove_margins()
        
        # Apply compensation
        self.compensation = fk.Matrix(spill, detectors, fluorochromes)
        self._comp_events = self.compensation.apply(self)
        self.tube = self.original_filename.split("_")[1].split(" ")[-1]
        
        self.apply_transform(logicle_xform)

        # Reset subsample indices to account for removed clipped rows
        if "subsample" in kwargs.keys():
            subsample = kwargs["subsample"]
        else:
            subsample = None
        self.subsample_events(subsample)

    def _remove_margins(self):

        # TODO Do we need margins removal channels which are not SS and FS?
        raw_df = self.as_dataframe(source="raw")
        clipped = raw_df.apply(get_clipped_idx, axis=0, raw=True)
        clipped_rows = np.where(clipped,)[0]
        self._raw_events = raw_df.drop(clipped_rows, axis=0).values



def lazy_fcs_loading(datadir, is_preproc=True, *sample_args, **sample_kwargs):
    SampleClass = PreprocSample if is_preproc is True else CustomSample
    for fpath in Path(datadir).iterdir():
        sample = SampleClass(fpath, *sample_args, **sample_kwargs)
        yield sample