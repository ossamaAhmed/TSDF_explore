from TSDF_explore.models.state_encoder_decoder_v1 import StateEncoderDecoder
import torch
import os


class ModelLoader(object):
    def __init__(self, model_class, trained_model_path, gpu=True):
        self.model = None
        if model_class == "state_encoder_v1":
            if gpu:
                self.model = StateEncoderDecoder().cuda()
            else:
                self.model = StateEncoderDecoder()
        else:
            raise Exception("Model Class is not defined")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not gpu:
            self.model.load_state_dict(torch.load(os.path.join(current_dir, "../", trained_model_path),
                                                  map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(os.path.join(current_dir, "../", trained_model_path)))
        # self.model.eval()
        return

    def get_inference_model(self):
        return self.model