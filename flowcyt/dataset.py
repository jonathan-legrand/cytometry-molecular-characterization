from pathlib import Path
import pandas as pd
import numpy as np
import flowkit as fk
from flowcyt.loading import PreprocSample
from torch.utils.data import Dataset, Subset
from flowcyt.utils import get_config
import torch
import numpy as np
import joblib
import csv
from sklearn.base import BaseEstimator

config = get_config()
MPATH = config["META_PATH"]
LABEL_MAPPING = {
            "WT": 0,
            "NF": 0,
            "ITD": 1,
            "ITD + TKD": 1,
            "ITD+TKD": 1,
            "TKD": 0,
            "NF (abs de dossier glims)": 0
        }

NPM_MAPPING = {'neg': 0,
 'NPM type D': 1,
 'NPM type A': 1,
 'NPM type B': 1,
 'NPM1 muté': 1,
 'NPM muté type A': 1,
 'NPM1 muté type Nm': 1,
 'NPM1 muté type A': 1,
 'NPM type Ht': 1,
 'NPM type Om non suivable': 1,
 'NPM1 muté type Z (pas de suivi MRD possible bio mol)': 1,
 'muté type 4/Ht': 1,
 'muté type A': 1
}

one_hot_sex_encoding = {"1": [1, 0], "2": [0, 1]}

TUBES = ("A", "B", "C")

tabular_features = (
    "Blastes moelle osseuse (%)",
    "age",
    "sexe"
)
ELN_features = (
    "FS-A",
    "SS-A",
    "CD34 PC7",
    "CD13 ECD", # Tube A
    "CD38 PB", # Tube B
    "CD7 APC700", # Tube A
    "CD33 PC5.5",
    "CD56 APC700", # Tube B
    "CD117 APC", # Tube A
    "HLA-DR PB", # Tube C
    "CD45 KO"
)

class CytometryDataset(Dataset):
    def __init__(
            self,
            dpath:str,
            mpath:str=MPATH,
            scaler:BaseEstimator=None,
            n_cells:int=5000,
            resample_cells:bool=False,
            random_state:int=1234,
            return_patient_id:bool=False,
            tabular_features:tuple=None,
            target_col:str="FLT3",
            tubes:tuple=TUBES,
            filter_npm=False
        ):
        if isinstance(dpath, str):
            dpath = Path(dpath)
        self.fc_dataset = self._build_fc_dataset(dpath, mpath, n_cells)
        self.n_cells = n_cells
        self.rng = np.random.default_rng(random_state)
        self.scaler = scaler
        self.resample_cells = resample_cells
        self.return_patient_id = return_patient_id
        self.tabular_features = tabular_features
        self.target_col = target_col
        self.tubes = tubes

        if target_col == "FLT3":
            self.label_mapping = LABEL_MAPPING
        elif target_col == "NPM":
            self.label_mapping = NPM_MAPPING
        else:
            raise NotImplementedError("Unsupported target column")

        print("before", self.__len__())
        self._validate_n_tubes()
        self._drop_nf(target_col)
        if filter_npm:
            self._filter_npm()
        print("after", self.__len__())

        

    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_excel(mpath)
        self.metadata = metadata
        patients = metadata.Patient.to_list()
        fc_dataset = {patient_id: {} for patient_id in patients}
        # Build frame
        for path in dpath.iterdir():
            sample = PreprocSample(path, subsample=n_cells)
            fc_dataset[sample.patient][sample.tube] = sample
        return fc_dataset

    def _validate_n_tubes(self):
        """
        Check number of tubes for each patient and
        remove invalid entries.
        """
        patients = self.metadata.Patient.to_list()
        for patient in patients:
            patient_keys = self.fc_dataset[patient].keys()
            missing_tubes = [tube not in patient_keys for tube in self.tubes]
            if any(missing_tubes):
                print(f"{patient} has {sum(missing_tubes)} missing tubes, removing")
                self.fc_dataset.pop(patient)
                drop_idx = self.metadata[self.metadata.Patient == patient].index
                self.metadata.drop(index=drop_idx, inplace=True)

    def _drop_nf(self, target_col):
        for _, row in self.metadata.iterrows():
            target = row[target_col]
            if "NF" in target or "NR" in target:
                patient = row.Patient
                print(f"{patient} has NF status, removing")
                self._delete_patient(patient)
                
    def _delete_patient(self, patient):
        self.fc_dataset.pop(patient)
        drop_idx = self.metadata[self.metadata.Patient == patient].index
        self.metadata.drop(index=drop_idx, inplace=True)

    def _filter_npm(self):
        """
        Experiment to check that FLT3 / CD34- asssociation
        """
        # We want to remove NF for NPM even though FLT3 is the target
        self._drop_nf("NPM")
        for _, row in self.metadata.iterrows():
            if NPM_MAPPING[row.NPM] == 1:
                self._delete_patient(row.Patient)

    def __getitem__(self, index):
        features, label, row = self.get_numpy(index)

        features = torch.tensor(features, dtype=torch.get_default_dtype())
        label = torch.tensor(label, dtype=torch.get_default_dtype())
        output = [features, label]
        if self.tabular_features is not None:
            tabular_data = self.get_tabular(row)
            output.append(
                torch.tensor((tabular_data), dtype=torch.get_default_dtype())
            )

        if self.return_patient_id:
            output.append(row.Patient)
        return output
    
    def get_tabular(self, row):
        tabular_data = []
        for feature in self.tabular_features:

            tabular_value = row[feature]
            if feature == "sexe":
                tabular_value = one_hot_sex_encoding[str(tabular_value)]
                tabular_data += tabular_value
            else:
                tabular_data.append(tabular_value)

        return np.stack(tabular_data) # Follow sklearn convention

    def get_df(self, index):

        row = self.metadata.iloc[index, :]
        patient_samples = self.fc_dataset[row.Patient]
        features = []
        for tube in self.tubes:
            sample = patient_samples[tube]
            indices = np.arange(sample.raw_shape[0])

            if self.resample_cells:
                sampled_indices = self.rng.choice(
                    indices, size=self.n_cells, replace=False
                )
                event_mask = np.where(np.isin(indices, sampled_indices), True, False)
                subsample = False
            else:
                event_mask = None
                subsample = True

            events = sample.as_dataframe(
                source="raw",
                subsample=subsample,
                col_multi_index=False,
                col_names=sample.pns_labels,
                event_mask=event_mask
            ).drop(
                ["TIME", " ", "", "FS PEAK LIN", "FS-H"], axis=1, errors="ignore"
            )
            events.rename({"CD38PB":"CD38 PB"}, axis=1, inplace=True)
            if (len(events.columns) != 12):
                raise ValueError(f"{row.Patient} tube {tube} has {len(events.columns)} columns")
            
            events = events.loc[:, pns_dict_camilla[tube]] # Make sure we always have the correct order
            
            features.append(
                events
            )
        label = self.get_label(row)
        
        return features, label, row

    def get_numpy(self, index):
        features, label, row = self.get_df(index)
        try:
            features = np.stack([f.to_numpy() for f in features])
        except ValueError as err:
            print(features)
            print([f.shape for f in features])
            raise err
        if self.scaler is not None:
            features = self.scaler.transform(features).squeeze()
        return features, label, row

    def get_label(self, row):
        return self.label_mapping[row[self.target_col]]

    def __len__(self):
        return len(self.fc_dataset)
    
    def fit_scaler(self, ScalerClass):
        X = [self.__getitem__(i)[0] for i in range(self.__len__())]
        self.scaler = ScalerClass().fit(np.stack(X))
        return self

    def set_scaler(self, scaler):
        """
        We use a setter func to avoid setting the scaler
        on the torch subset rather than on the actual dataset.
        """
        self.scaler = scaler
        return self


class CytoSubset(Subset):
    scaler = None
    __getitems__ = None

    def __getitem__(self, idx):
        elements = super().__getitem__(idx)
        if self.scaler is not None:
            if self.tabular_features is not None:
                cyto_scaled, tabular_scaled = self.scaler.transform(
                    elements[0],
                    tabular_data=elements[2]
                )
                output = [cyto_scaled, elements[1], tabular_scaled]
                if self.return_patient_id:
                    output.append(elements[-1])
            else:
                cyto_scaled = self.scaler.transform(elements[0])
                output = [cyto_scaled, *elements[1:]]
            return output
        else:
            return elements

                

    def fit_scaler(self, ScalerClass):
        X = []
        tabular = []
        if self.tabular_features is not None:
            scaler = ScalerClass(include_tabular=True)
            for i in range(len(self)):
                items = self[i]
                X.append(items[0])
                tabular.append(items[2])
            tabular = np.stack(tabular)
        else:
            scaler = ScalerClass(include_tabular=False)
            X = [self[i][0] for i in range(len(self))] # TODO Why did it escape tests
            tabular = None

        self.scaler = scaler.fit(
            np.stack(X), tabular_data=tabular
        )
        return self

    def set_scaler(self, scaler):
        """
        We use a setter func to avoid setting the scaler
        in the wrong place. Well it should be fancy property stuff
        but whatever
        """
        self.scaler = scaler
        return self
    def store_scaler(self, path):
        return joblib.dump(self.scaler, path)

    def __getattr__(self, attr):
        # Prevent recursion issues for special attributes
        if attr in ("__dict__", "__setstate__", "__getstate__"):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{attr}'")

        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    @property
    def resample_cells(self):
        return self.dataset.resample_cells

    @resample_cells.setter
    def resample_cells(self, truth_value: bool):
        assert isinstance(truth_value, bool)
        self.dataset.resample_cells = truth_value

pns_dict_camilla = {'A': ['FS-A',
  'SS-A',
  'CD65 FITC',
  'CD14 PE',
  'CD13 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD117 APC',
  'CD7 APC700',
  'CD11b APC750',
  'CD16 PB',
  'CD45 KO'],
 'B': ['FS-A',
  'SS-A',
  'CD64 FITC',
  'CD10 PE',
  'CD4 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD123 APC',
  'CD56 APC700',
  'CD19 APC750',
  'CD38 PB',
  'CD45 KO'],
 'C': ['FS-A',
  'SS-A',
  'CD36 FITC',
  'CD61 PE',
  'ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD2 APC',
  'APC700',
  'CD71 APC750',
  'HLA-DR PB',
  'CD45 KO']}


pns_dict = {'A': ['FS-A',
  'SS-A',
  'CD65 FITC',
  'CD14 PE',
  'CD13 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD117 APC',
  'CD7 APC700',
  'CD11b APC750',
  'CD16 PB',
  'CD45 KO'],
 'B': ['FS-A',
  'SS-A',
  'CD64 FITC',
  'CD10 PE',
  'CD4 ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD123 APC',
  'CD56 APC700',
  'CD19 APC750',
  'CD38 PB',
  'CD45 KO'],
 'C': ['FS-A',
  'SS-A',
  'CD36 FITC',
  'CD61 PE',
  'ECD',
  'CD33 PC5.5',
  'CD34 PC7',
  'CD2 APC',
  'APC700',
  'CD71 APC750',
  'HLA-DR PB',
  'CD45 KO']}

def match_labels(pns_labels, tube):
    for label in pns_dict[tube]:
        if label not in pns_labels:
            return False
    return True


class TestSet(CytometryDataset):
    """
    Dataset for GEIL tubes that are used as a test set
    """
    def _build_fc_dataset(self, dpath, mpath, n_cells):
        metadata = pd.read_excel(mpath)
        metadata.rename(columns={"Identifiant patient":"Patient"}, inplace=True)
        metadata.Patient = metadata.Patient.astype(str)
        metadata["FLT3"] = np.where(metadata["Mutation FLT3-ITD"] == "Oui", "ITD", "WT")
        metadata["NPM"] = np.where(
            metadata["Mutation de NPM1 (Biologie moléculaire)"] == "Oui", "NPM1 muté", "neg"
        )

        self.metadata = metadata
        patients = metadata.Patient.to_list()
        fc_dataset = {patient_id: {} for patient_id in patients}
        
        # Initialize excluded patients file
        excluded_file = Path("output/excluded_patient.csv")
        excluded_file.parent.mkdir(parents=True, exist_ok=True)
        with open(excluded_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Patient_ID", "Tube", "PNS_Labels"])

        for fcs_path in dpath.iterdir():
            sample = fk.Sample(
                str(fcs_path),
                subsample=n_cells,
                null_channel_list=["FS PEAK", "FS-H", "FS PEAK LIN", ""]
            )
            if sample.pns_labels[2] == "-FITC":
                print(sample.original_filename, " has truncated pns names, skipping")
                continue
            
            # Make sure we use FS-A and SS-A for compatibility with other pipelines
            sample.pns_labels = [x.replace(" INT LIN", "-A") for x in sample.pns_labels]
            sample.pns_labels = [x.replace("CD38PB", "CD38 PB") for x in sample.pns_labels]

            sample.raw_shape = sample._raw_events.shape
            tags = sample.original_filename.split("_")
            patient_id = tags[0]
            tube = tags[1][4]
            assert tube in {"A", "B", "C"}, tube
            
            if not match_labels(sample.pns_labels, tube):
                print(sample.original_filename, "Tube does not have required pns labels, skipping")
                # Append excluded sample to CSV
                with open("output/excluded_patient.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([patient_id, tube, ";".join(sample.pns_labels)])
                continue

            fc_dataset[patient_id][tube] = sample
        return fc_dataset


def fetch_X_y_meta(dataset, raise_shape=True):
    X, y, metadata = [], [], []
    for i in range(len(dataset)):
        try:
            patient_data = dataset.get_numpy(i)
        except ValueError as err:
            if raise_shape:
                print(f"Patient {i} has not enough cells")
                raise err
            else:
                print(f"Patient {i} has not enough cells, skipping")
                continue

        X.append(patient_data[0])
        y.append(patient_data[1])
        metadata.append(patient_data[2])

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    metadata = pd.DataFrame(metadata)
    return X, y, metadata