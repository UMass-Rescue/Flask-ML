from fastai.vision import *
import json

class ImageNetClassifier:
    def __init__(self):
        self.model = models.resnet34(pretrained=True).cuda()

    def predict(self, img:'np.ndarray of shape (H, W, 3)'):
        res = self.model(torch.tensor(img)[None].cuda())
        res = res.detach().cpu().numpy().flatten().tolist()
        return json.dumps(res)