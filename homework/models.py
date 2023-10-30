import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    # Convert heatmap to 4D tensor
    heatmap_4d = heatmap[None, None]
    
    # Apply max pooling
    pooled = F.max_pool2d(heatmap_4d, max_pool_ks, stride=1, padding=max_pool_ks//2)
    
    # Convert back to 2D
    pooled = pooled[0, 0]
    
    # Find local maxima
    peaks = (heatmap == pooled) & (heatmap > min_score)
    
    # Get coordinates and scores of local maxima
    y, x = torch.nonzero(peaks, as_tuple=True)
    scores = heatmap[peaks]
    
    # Sort scores in descending order and take top max_det detections
    _, idx = scores.sort(descending=True)
    idx = idx[:max_det]
    
    # Get final coordinates and scores
    final_y = y[idx]
    final_x = x[idx]
    final_scores = scores[idx]
    
    # Convert to list of tuples
    result = [(s.item(), x.item(), y.item()) for s, x, y in zip(final_scores, final_x, final_y)]
    
    return result

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Detector(nn.Module):
    def __init__(self, num_classes=3, output_channels=3):
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.output_channels = output_channels

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128, 256, stride=2)
        self.fc = nn.Conv2d(256, output_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        
        # Upsample to match the target tensor's shape
        x = F.interpolate(x, size=(96, 128), mode='bilinear', align_corners=False)
        
        heatmaps = self.fc(x)
        return heatmaps

    def detect(self, image):
        """
        @image: 3 x H x W image
        @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                return no more than 30 detections per image per class. You only need to predict width and height
                for extra credit. If you do not predict an object size, return w=0, h=0.
        """
        # Convert the image to a PyTorch tensor
        image = image.clone().detach().to(dtype=torch.float32)
        if len(image.shape) == 3:  # Add batch dimension if missing
            image = image.unsqueeze(0)

        # Forward pass to get the predicted heatmaps
        predicted_heatmaps = self.forward(image)
        
        # Initialize empty lists to store detections for each class
        kart_detections = []
        bomb_detections = []
        pickup_detections = []
        
        # Extract peaks from the predicted heatmaps
        for i, label in enumerate(['kart', 'bomb', 'pickup']):
            heatmap = predicted_heatmaps[0, i]
            peaks = extract_peak(heatmap)
            
            # Populate the detections list for each class
            detections_list = kart_detections if label == 'kart' else bomb_detections if label == 'bomb' else pickup_detections
            for peak in peaks:
                score, x, y = peak
                detections_list.append((score, x, y, 0, 0))
        
        # Limit the number of detections to 30 for each class
        kart_detections = sorted(kart_detections, reverse=True)[:30]
        bomb_detections = sorted(bomb_detections, reverse=True)[:30]
        pickup_detections = sorted(pickup_detections, reverse=True)[:30]

        # Return detections as a tuple of lists
        return kart_detections, bomb_detections, pickup_detections


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
