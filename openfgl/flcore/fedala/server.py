from openfgl.flcore.fedavg.server import FedAvgServer

class FedALAServer(FedAvgServer):
    """
    FedALAServer is functionally identical to FedAvgServer.
    FedALA adapts the model on the client side after downloading the global model.
    The server simply aggregates uploaded models via weighted averaging.
    """
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)