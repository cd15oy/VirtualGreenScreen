import torch
import torchvision 
import numpy as np

class Segmenter:
    def __init__(self):
        self.imgTransform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.threshold = 0.5
        
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.cuda().eval()

    def pred(self,img):
        # Define the standard transforms that need to be done at inference time
        
        out = self.model(self.imgTransform(img).unsqueeze(0).cuda())
        output = out["out"][0]
        output = output.argmax(0)
            
        w,h = output.shape
        true = output.new_full((w,h), 1.0, dtype=torch.float)
        false = output.new_full((w,h), 0.0, dtype=torch.float)
            
            
        out = torch.where(output == 15, true, false)
        out = out.cpu().numpy()
        return out