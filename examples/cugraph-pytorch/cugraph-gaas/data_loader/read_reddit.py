# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
from gaas_client import GaasClient

from dgl.contrib.cugraph import GaasGraphStorage


def read_reddit(dataset_dir, self_loop=False):
    """
    Reads the reddit dataset files from dataset_dir and returns a
    GaasGraphStorage instance, labels, and data mask arrays for various training
    node types.
    """
    dataset_path_obj = Path(dataset_dir).absolute()

    # Use the defaults for host and port (localhost, 9090)
    # Assume the server is running and using the same defaults!
    gaas_client = GaasClient()

    # Call the user-defined extension loaded by the GaaS server for reading and
    # cleaning the reddit data and returning an ID to a server-side cugraph
    # PropertyGraph instance. That instance is required for constructing a
    # client-side GaaSGraphStorage instance.
    graph_id = gaas_client.call_graph_creation_extension(
        "create_property_graph_from_reddit_data",
        str(dataset_path_obj/"reddit_graph.npz"),
        str(dataset_path_obj/"reddit_data.npz"))
    print(f"----> {graph_id=}")

    gstore = GaasGraphStorage(gaas_client, graph_id)

    # FIXME: ideally this would not need to be read here if it was already read
    # once by the GaaS server.
    reddit_data = np.load(dataset_path_obj/"reddit_data.npz")

    labels = reddit_data["label"]
    # train/val/test indices
    node_types = reddit_data["node_types"]
    train_mask = (node_types == 1)
    val_mask = (node_types == 2)
    test_mask = (node_types == 3)

    return gstore, labels, train_mask, val_mask, test_mask
