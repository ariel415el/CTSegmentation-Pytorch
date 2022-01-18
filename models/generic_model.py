
class SegmentationModel:
    def __init__(self, n_channels, n_classes, device):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device

    def train_one_sample(self, ct_volume, gt_volume, mask, global_step):
        """
        Train this model on a single sample.
        Returns a dictionary of loss values.
        """
        raise NotImplementedError()

    def step_scheduler(self, evaluation_score):
        """
        step the learning rate scheduler
        """
        raise NotImplementedError()

    def predict_volume(self, ct_volume):
        """
        Return the prediciton of the model on some input
        """
        raise NotImplementedError()

    def get_state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()
