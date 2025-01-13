"""
Class for loading BioBridgePrimeKG dataset.
"""

import os
import pickle
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from .dataset import Dataset
from .primekg import PrimeKG

class BioBridgePrimeKG(Dataset):
    """
    Class for loading BioBridgePrimeKG dataset.
    It downloads the data from the BioBridge repo and stores it in the local directory.
    The data is then loaded into pandas DataFrame of nodes and edges.
    This class was adapted from the BioBridge repo:
    https://github.com/RyanWangZf/BioBridge
    """

    def __init__(self,
                 primekg_dir: str = "../../../data/primekg/",
                 local_dir: str = "../../../data/biobridge_primekg/",
                 random_seed: int=0,
                 n_neg_samples: int=5):
        """
        Constructor for BioBridgePrimeKG class.

        Args:
            primekg_dir (str): The directory of PrimeKG dataset.
            local_dir (str): The directory to store the downloaded data.
            random_seed (int): The random seed value.
        """
        self.name: str = "biobridge_primekg"
        self.primekg_dir: str = primekg_dir
        self.local_dir: str = local_dir
        self.random_seed = random_seed
        self.n_neg_samples = n_neg_samples
        # Preselected node types:
        # protein, molecular function, cellular component, biological process, drug, disease
        self.preselected_node_types = ["protein", "mf", "cc", "bp", "drug", "disease"]
        self.node_type_map = {
            "protein": "gene/protein",
            "mf": "molecular_function",
            "cc": "cellular_component",
            "bp": "biological_process",
            "drug": "drug",
            "disease": "disease",
        }

        # Attributes to store the data
        self.primekg = None
        self.primekg_triplets = None
        self.primekg_triplets_negative = None
        self.data_config = None
        self.emb_dict = None
        self.df_train = None
        self.df_node_train = None
        self.df_test = None
        self.df_node_test = None
        self.node_info_dict = None

        # Set up the dataset
        self.setup()

    def setup(self):
        """
        A method to set up the dataset.
        """
        # Make the directories if it doesn't exist
        os.makedirs(os.path.dirname(self.primekg_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.local_dir), exist_ok=True)

        # Set the random seed
        self.set_random_seed(self.random_seed)

        # Set SettingWithCopyWarning  warnings to none
        pd.options.mode.chained_assignment = None

    def _load_primekg(self) -> PrimeKG:
        """
        Private method to load related files of PrimeKG dataset.

        Returns:
            The PrimeKG dataset.
        """
        primekg_data = PrimeKG(local_dir=self.primekg_dir)
        primekg_data.load_data()

        return primekg_data

    def _download_file(self,
                       remote_url:str,
                       local_dir: str,
                       local_filename: str):
        """
        A helper function to download a file from remote URL to the local directory.

        Args:
            remote_url (str): The remote URL of the file to be downloaded.
            local_dir (str): The local directory to store the downloaded file.
            local_filename (str): The local filename to store the downloaded file.
        """
        # Make the local directory if it does not exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        # Download the file from remote URL to local directory
        local_path = os.path.join(local_dir, local_filename)
        if os.path.exists(local_path):
            print(f"File {local_filename} already exists in {local_dir}.")
        else:
            print(f"Downloading {local_filename} from {remote_url} to {local_dir}...")
            response = requests.get(remote_url, stream=True, timeout=300)
            response.raise_for_status()
            progress_bar = tqdm(
                total=int(response.headers.get("content-length", 0)),
                unit="iB",
                unit_scale=True,
            )
            with open(os.path.join(local_dir, local_filename), "wb") as file:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

    def _load_data_config(self) -> dict:
        """
        Load the data config file of BioBridgePrimeKG dataset.

        Returns:
            The data config file of BioBridgePrimeKG dataset.
        """
        # Download the data config file of BioBridgePrimeKG
        self._download_file(
            remote_url= ('https://raw.githubusercontent.com/RyanWangZf/BioBridge/'
                         'refs/heads/main/data/BindData/data_config.json'),
            local_dir=self.local_dir,
            local_filename='data_config.json')

        # Load the downloaded data config file
        with open(os.path.join(self.local_dir, 'data_config.json'), 'r', encoding='utf-8') as f:
            data_config = json.load(f)

        return data_config

    def _build_node_embeddings(self) -> dict:
        """
        Build the node embeddings for BioBridgePrimeKG dataset.

        Returns:
            The dictionary of node embeddings.
        """
        processed_file_path = os.path.join(self.local_dir, "embeddings", "embedding_dict.pkl")
        if os.path.exists(processed_file_path):
            # Load the embeddings from the local directory
            with open(processed_file_path, "rb") as f:
                emb_dict_all = pickle.load(f)
        else:
            # Download the embeddings from the BioBridge repo and further process them
            # List of embedding source files
            url = ('https://media.githubusercontent.com/media/RyanWangZf/BioBridge/'
                   'refs/heads/main/data/embeddings/esm2b_unimo_pubmedbert/')
            file_list = [f"{n}.pkl" for n in self.preselected_node_types]

            # Download the embeddings
            for file in file_list:
                self._download_file(remote_url=os.path.join(url, file),
                                    local_dir=os.path.join(self.local_dir, "embeddings"),
                                    local_filename=file)

            # Unified embeddings
            emb_dict_all = {}
            for file in file_list:
                with open(os.path.join(self.local_dir, "embeddings", file), "rb") as f:
                    emb = pickle.load(f)
                emb_ar = emb["embedding"]
                if not isinstance(emb_ar, list):
                    emb_ar = emb_ar.tolist()
                emb_dict_all.update(dict(zip(emb["node_index"], emb_ar)))

            # Store embeddings
            with open(processed_file_path, "wb") as f:
                pickle.dump(emb_dict_all, f)

        return emb_dict_all

    def _build_full_triplets(self) -> tuple[pd.DataFrame, dict]:
        """
        Build the full triplets for BioBridgePrimeKG dataset.

        Returns:
            The full triplets for BioBridgePrimeKG dataset.
            The dictionary of node information.
        """
        processed_file_path = os.path.join(self.local_dir, "processed", "triplet_full.tsv.gz")
        if os.path.exists(processed_file_path):
            # Load the file from the local directory
            with open(processed_file_path, "rb") as f:
                primekg_triplets = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)

            # Load each dataframe in the local directory
            node_info_dict = {}
            for i, node_type in enumerate(self.preselected_node_types):
                with open(os.path.join(self.local_dir, "processed",
                                       f"{node_type}.csv"), "rb") as f:
                    df_node = pd.read_csv(f)
                node_info_dict[self.node_type_map[node_type]] = df_node
        else:
            # Download the related files from the BioBridge repo and further process them
            # List of processed files
            url = ('https://media.githubusercontent.com/media/RyanWangZf/BioBridge/'
                   'refs/heads/main/data/Processed/')
            file_list = ["protein", "molecular", "cellular", "biological", "drug", "disease"]

            # Download the processed files
            for i, file in enumerate(file_list):
                self._download_file(remote_url=os.path.join(url, f"{file}.csv"),
                                    local_dir=os.path.join(self.local_dir, "processed"),
                                    local_filename=f"{self.preselected_node_types[i]}.csv")

            # Build the node index list
            node_info_dict = {}
            node_index_list = []
            for i, file in enumerate(file_list):
                df_node = pd.read_csv(os.path.join(self.local_dir, "processed",
                                                   f"{self.preselected_node_types[i]}.csv"))
                node_info_dict[self.node_type_map[self.preselected_node_types[i]]] = df_node
                node_index_list.extend(df_node["node_index"].tolist())

            # Filter the PrimeKG dataset to take into account only the selected node types
            primekg_triplets = self.primekg.get_edges().copy()
            primekg_triplets = primekg_triplets[
                primekg_triplets["head_index"].isin(node_index_list) &\
                primekg_triplets["tail_index"].isin(node_index_list)
            ]
            primekg_triplets = primekg_triplets.reset_index(drop=True)

            # Perform mapping of node types
            primekg_triplets["head_type"] = primekg_triplets["head_type"].apply(
                lambda x: self.data_config["node_type"][x]
            )
            primekg_triplets["tail_type"] = primekg_triplets["tail_type"].apply(
                lambda x: self.data_config["node_type"][x]
            )

            # Perform mapping of relation types
            primekg_triplets["display_relation"] = primekg_triplets["display_relation"].apply(
                lambda x: self.data_config["relation_type"][x]
            )

            # Store the processed triplets
            primekg_triplets.to_csv(processed_file_path, sep="\t", compression="gzip", index=False)

        return primekg_triplets, node_info_dict

    def _build_train_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build the train-test split for BioBridgePrimeKG dataset.

        Returns:
            The train triplets for BioBridgePrimeKG dataset.
            The train nodes for BioBridgePrimeKG dataset.
            The test triplets for BioBridgePrimeKG dataset.
            The test nodes for BioBridgePrimeKG dataset.
            The full triplets for BioBridgePrimeKG dataset.
        """
        if os.path.exists(os.path.join(self.local_dir, "processed",
                                       "triplet_full_altered.tsv.gz")):
            # Load each dataframe in the local directory
            with open(os.path.join(self.local_dir, "processed",
                                   "triplet_train.tsv.gz"), "rb") as f:
                df_train = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)

            with open(os.path.join(self.local_dir, "processed",
                                   "node_train.tsv.gz"), "rb") as f:
                df_node_train = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)

            with open(os.path.join(self.local_dir, "processed",
                                   "triplet_test.tsv.gz"), "rb") as f:
                df_test = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)

            with open(os.path.join(self.local_dir, "processed",
                                   "node_test.tsv.gz"), "rb") as f:
                df_node_test = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)

            with open(os.path.join(self.local_dir, "processed",
                                   "triplet_full_altered.tsv.gz"), "rb") as f:
                triplets = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)
        else:
            # Filtering out some nodes in the embedding dictionary
            triplets = self.primekg_triplets.copy()
            triplets = triplets[
                triplets["head_index"].isin(list(self.emb_dict.keys())) &\
                triplets["tail_index"].isin(list(self.emb_dict.keys()))
            ].reset_index(drop=True)

            # Perform splitting of the triplets
            list_split = {
                "train": [],
                "test": [],
            }
            node_split = {
                "train": {
                    "node_index": [],
                    "node_type": [],
                },
                "test": {
                    "node_index": [],
                    "node_type": [],
                }
            }
            # Loop over the node types
            for node_type in triplets["head_type"].unique():
                df_sub = triplets[triplets["head_type"] == node_type]
                all_x_indexes = df_sub["head_index"].unique()
                # By default, we use 90% of the nodes for training and 10% for testing
                te_x_indexes = np.random.choice(
                    all_x_indexes, size=int(0.1*len(all_x_indexes)), replace=False
                )
                df_subs = {}
                df_subs["test"] = df_sub[df_sub["head_index"].isin(te_x_indexes)]
                df_subs["train"] = df_sub[~df_sub["head_index"].isin(te_x_indexes)]
                list_split["train"].append(df_subs["train"])
                list_split["test"].append(df_subs["test"])

                # record the split
                node_index = {}
                node_index["train"] = df_subs["train"]["head_index"].unique()
                node_split["train"]["node_index"].extend(node_index["train"].tolist())
                node_split["train"]["node_type"].extend([node_type]*len(node_index["train"]))
                node_index["test"] = df_subs["test"]["head_index"].unique()
                node_split["test"]["node_index"].extend(node_index["test"].tolist())
                node_split["test"]["node_type"].extend([node_type]*len(node_index["test"]))

                print(f"Number of {node_type} nodes in train: {len(node_index['train'])}")
                print(f"Number of {node_type} nodes in test: {len(node_index['test'])}")

            # Prepare train and test DataFrames
            df_train = pd.concat(list_split["train"])
            df_node_train = pd.DataFrame(node_split["train"])
            df_test = pd.concat(list_split["test"])
            df_node_test = pd.DataFrame(node_split["test"])

            # Store each dataframe in the local directory
            df_train.to_csv(os.path.join(self.local_dir, "processed", "triplet_train.tsv.gz"),
                            sep="\t", compression="gzip", index=False)
            df_node_train.to_csv(os.path.join(self.local_dir, "processed", "node_train.tsv.gz"),
                                sep="\t", compression="gzip", index=False)
            df_test.to_csv(os.path.join(self.local_dir, "processed", "triplet_test.tsv.gz"),
                           sep="\t", compression="gzip", index=False)
            df_node_test.to_csv(os.path.join(self.local_dir, "processed", "node_test.tsv.gz"),
                                sep="\t", compression="gzip", index=False)
            # Store altered full triplets as well
            triplets.to_csv(os.path.join(self.local_dir, "processed",
                                         "triplet_full_altered.tsv.gz"),
                            sep="\t", compression="gzip", index=False)

        return df_train, df_node_train, df_test, df_node_test, triplets

    # def _negative_sampling(self,
    #                        batch_df: pd.DataFrame,
    #                        process_index: int,
    #                        index_map: dict,
    #                        node_train_dict: dict) -> pd.DataFrame:
    #     """
    #     A helper function to perform negative sampling for a batch of triplets.
    #     """
    #     negative_y_index_list = []
    #     for _, row in tqdm(batch_df.iterrows(),
    #                        total=batch_df.shape[0],
    #                        desc=f"Process {process_index}"):
    #         x_index = row['head_index']
    #         # y_index = row['y_index']
    #         y_index_type = row['tail_type']
    #         paired_y_index_list = index_map[x_index]

    #         # sample a list of negative y_index
    #         node_train_sub = node_train_dict[y_index_type]
    #         negative_y_index = node_train_sub[
    #             ~node_train_sub['node_index'].isin(paired_y_index_list)
    #         ]['node_index'].sample(self.n_neg_samples).tolist()
    #         negative_y_index_list.append(negative_y_index)

    #     batch_df.loc[:, 'negative_tail_index'] = negative_y_index_list
    #     return batch_df

    # def _build_negative_triplets(self,
    #                              chunk_size: int=100000,
    #                              n_neg_samples: int=10):
    #     """
    #     Build the negative triplets for BioBridgePrimeKG dataset.
    #     """
    #     processed_file_path = os.path.join(self.local_dir,
    #                                        "processed",
    #                                        "triplet_train_negative.tsv.gz")
    #     if os.path.exists(processed_file_path):
    #         # Load the negative triplets from the local directory
    #         with open(processed_file_path, "rb") as f:
    #             triplets_negative = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False)
    #     else:
    #         # Set the number samples for negative sampling
    #         self.n_neg_samples = n_neg_samples

    #         # Split node list by type
    #         node_train_dict = {}
    #         type_list = self.df_node_train['node_type'].unique()
    #         for node_type in type_list:
    #             node_train_dict[node_type] = self.df_node_train[
    #                 self.df_node_train['node_type'] == node_type
    #             ].reset_index(drop=True)

    #         # create an index mapping from x_index to y_index
    #         index_map = self.df_train[
    #             ['head_index', 'tail_index']
    #         ].drop_duplicates().groupby('head_index').agg(list).to_dict()['tail_index']

    #         # Negative sampling
    #         batch_df_list = []
    #         for i in tqdm(range(0, self.df_train.shape[0], chunk_size)):
    #             batch_df_list.append(self.df_train.iloc[i:i+chunk_size])
    #         # Process negative sampling
    #         results = [
    #             self._negative_sampling(batch_df,
    #                                     num_piece,
    #                                     index_map,
    #                                     node_train_dict)
    #                                     for num_piece, batch_df in enumerate(batch_df_list)
    #         ]

    #         # Store the negative triplets
    #         triplets_negative = pd.concat(results, axis=0)
    #         triplets_negative.to_csv(processed_file_path,
    #                                  sep="\t", compression="gzip", index=False)

    #     # Set attribute
    #     self.primekg_triplets_negative = triplets_negative

    #     return triplets_negative

    # def load_data(self,
    #               build_neg_triplest: bool= False,
    #               chunk_size: int=100000,
    #               n_neg_samples: int=10):

    def load_data(self):
        """
        Load the BioBridgePrimeKG dataset into pandas DataFrame of nodes and edges.

        Args:
            build_neg_triplest (bool): Whether to build negative triplets.
            chunk_size (int): The chunk size for negative sampling.
            n_neg_samples (int): The number of negative samples for negative sampling.
        """
        # Load PrimeKG dataset
        print("Loading PrimeKG dataset...")
        self.primekg = self._load_primekg()

        # Load data config file of BioBridgePrimeKG
        print("Loading data config file of BioBridgePrimeKG...")
        self.data_config = self._load_data_config()

        # Build node embeddings
        print("Building node embeddings...")
        self.emb_dict = self._build_node_embeddings()

        # Build full triplets
        print("Building full triplets...")
        self.primekg_triplets, self.node_info_dict = self._build_full_triplets()

        # Build train-test split
        print("Building train-test split...")
        self.df_train, self.df_node_train, self.df_test, self.df_node_test, self.primekg_triplets =\
        self._build_train_test_split()

        # if build_neg_triplest:
        #     # Build negative triplets
        #     print("Building negative triplets...")
        #     self.primekg_triplets_negative = self._build_negative_triplets(
        #         chunk_size=chunk_size,
        #         n_neg_samples=n_neg_samples
        #     )

    def set_random_seed(self, seed: int):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        np.random.seed(seed)

    def get_primekg(self) -> PrimeKG:
        """
        Get the PrimeKG dataset.

        Returns:
            The PrimeKG dataset.
        """
        return self.primekg

    def get_data_config(self) -> dict:
        """
        Get the data config file of BioBridgePrimeKG dataset.

        Returns:
            The data config file of BioBridgePrimeKG dataset.
        """
        return self.data_config

    def get_node_embeddings(self) -> dict:
        """
        Get the node embeddings for BioBridgePrimeKG dataset.

        Returns:
            The dictionary of node embeddings.
        """
        return self.emb_dict

    def get_primekg_triplets(self) -> pd.DataFrame:
        """
        Get the full triplets for BioBridgePrimeKG dataset.

        Returns:
            The full triplets for BioBridgePrimeKG dataset.
        """
        return self.primekg_triplets

    # def get_primekg_triplets_negative(self) -> pd.DataFrame:
    #     """
    #     Get the negative triplets for BioBridgePrimeKG dataset.

    #     Returns:
    #         The negative triplets for BioBridgePrimeKG dataset.
    #     """
    #     return self.primekg_triplets_negative

    def get_train_test_split(self) -> dict:
        """
        Get the train-test split for BioBridgePrimeKG dataset.

        Returns:
            The train-test split for BioBridgePrimeKG dataset.
        """
        return {
            "train": self.df_train,
            "node_train": self.df_node_train,
            "test": self.df_test,
            "node_test": self.df_node_test
        }

    def get_node_info_dict(self) -> dict:
        """
        Get the node information dictionary for BioBridgePrimeKG dataset.

        Returns:
            The node information dictionary for BioBridgePrimeKG dataset.
        """
        return self.node_info_dict
