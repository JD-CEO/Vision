# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision 
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import cv2
from tqdm.notebook import tqdm
import numpy as np

# %%
def load_specific_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None
    
def read_specific_line(file_path, line_num):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if line_num <= len(lines):
            return float(lines[line_num].strip())
        else:
            return None
        
def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


# %%
train_pth = "/kaggle/input/car-speeds/train.mp4"
train_trg_pth = "/kaggle/input/car-speeds/train.txt"
test_pth = "/kaggle/input/car-speeds/test.mp4"
test_trg_pth = "/kaggle/input/car-speeds/test.txt"

# %%
class Speed_Ds(Dataset):
    def  __init__(self, video_pth, target_pth, transform):
        super().__init__()
        self.video_pth, self.target_pth = video_pth, target_pth
        self.transform = transform
        
    def __len__(self):
        return get_num_frames(self.target_pth) - 1
    
    def __getitem__(self,idx):
        frame1 = torch.tensor(load_specific_frame(self.video_pth, idx)).permute((2,0,1))
        frame2 = torch.tensor(load_specific_frame(self.video_pth, idx + 1)).permute((2,0,1))
        target = torch.tensor([read_specific_line(self.target_pth, idx + 1)])
        return self.transform(frame1), self.transform(frame2), target

# %%
transform = transforms.Compose([
#     transforms.Lambda(lambda x: 2*(x/255)-1),
])

# %%
train_ds = Speed_Ds(train_pth, train_trg_pth, transform)
test_ds = Speed_Ds(test_pth, test_trg_pth, transform)

# %%
img_h, img_w = train_ds[0][0].shape[-2:]
img_h,img_w

# %%
fr1, fr2, trg = test_ds[0]

# %%
def optical_flow_batch(prev_frames, curr_frames):
    device = prev_frames.device
    prev_frames = prev_frames.cpu().numpy().transpose(0, 2, 3, 1)
    curr_frames = curr_frames.cpu().numpy().transpose(0, 2, 3, 1)
    
    prev_gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in prev_frames])
    curr_gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in curr_frames])
    
    flow = np.zeros((prev_frames.shape[0], prev_frames.shape[1], prev_frames.shape[2], 2), dtype=np.float32)
    for i in range(prev_frames.shape[0]):
        flow[i] = cv2.calcOpticalFlowFarneback(prev_gray[i], curr_gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_tensor = torch.from_numpy(flow).permute(0, 3, 1, 2).float().to(device)
    return flow_tensor

# %%
'''
    This one uses optical flow itself as its features
'''
class speed_flow(nn.Module):
    def __init__(self,img_h, img_w, kernel_size):
        super().__init__()
        self.cnv1 = nn.Conv2d(2, 32, kernel_size, padding=int((kernel_size - 1)//2))
        self.cnv2 = nn.Conv2d(32, 64, kernel_size, padding=int((kernel_size - 1)//2))
        self.cnv3 = nn.Conv2d(64, 16, 1)
        self.fc = nn.Linear(img_h*img_w*16, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self,fr1, fr2):
        B = fr1.shape[0]
        h = optical_flow_batch(fr1, fr2)
        h = self.tanh(self.cnv1(h))
        h = self.tanh(self.cnv2(h))
        h = self.relu(self.cnv3(h))
        return self.fc(h.reshape((B,-1)))
    

# %%
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.kernel_x = self.kernel_x.unsqueeze(0).to(torch.float32)
        self.kernel_y = self.kernel_y.unsqueeze(0).to(torch.float32)
    
    def forward(self, x):
        grad_x = F.conv2d(x, self.kernel_x.to(x.device).repeat(x.shape[-3],1,1).unsqueeze(1), padding=1, groups=x.shape[-3])
        grad_y = F.conv2d(x, self.kernel_y.to(x.device).repeat(x.shape[-3],1,1).unsqueeze(1), padding=1, groups=x.shape[-3])
        return torch.cat([grad_x, grad_y], dim=1)

# %%
'''
    This one uses features that optical flow itself has been created on
'''
class Motion_est(nn.Module):
    def __init__(self,img_h, img_w, kernel_size):
        super().__init__()
        self.sobel = Sobel()
        self.cnv1 = nn.Conv2d(15, 32, kernel_size, padding=int((kernel_size - 1)//2))
        self.cnv2 = nn.Conv2d(32, 64, kernel_size, padding=int((kernel_size - 1)//2))
        self.cnv3 = nn.Conv2d(64, 16, 1)
        self.fc = nn.Linear(img_h*img_w*16, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.GELU()
    
    def forward(self,fr1, fr2):
        B = fr1.shape[0]
        dt = fr2 - fr1
        dfr1 = self.sobel(fr1)
        dfr2 = self.sobel(fr2)
        h = torch.cat([dt, dfr1, dfr2], dim=1)
        h = self.tanh(self.cnv1(h))
        h = self.tanh(self.cnv2(h))
        h = self.relu(self.cnv3(h))
        return self.fc(h.view((B,-1)))
        

# %%
kernel_size = 7
lr = 0.1e-6
batch_size = 16
epochs = 10

# %%
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
model = Motion_est(img_h, img_w, kernel_size).to(device)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fun = torch.nn.L1Loss()

# %%
def learning(model, train, optimizer, loss_fun, device, epochs):
    t_losses = []
    
    for ep in range(epochs):
        model.train()
        pbar = tqdm(enumerate(train))
        for i, (fr1, fr2, trg) in pbar: # Y=<B, S, C>
            fr1 = fr1.to(device).to(torch.float32)
            fr2 = fr2.to(device).to(torch.float32)
            trg = trg.to(device).to(torch.float32)
            logits = model(fr1, fr2)

            loss = loss_fun(logits, trg) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_losses.append(loss.to("cpu").item())
            pbar.set_description(f"Epoch : {ep + 1} Training loss is => {loss} ")
            # del x, y
            torch.cuda.empty_cache()
    return model, t_losses

@torch.no_grad()
def evluate(model, test, loss_fun, device):
    t_loss = 0
    model.eval()
    pbar = tqdm(enumerate(test))
    for i, (fr1, fr2, trg) in pbar: # Y=<B, S, C>
        fr1 = fr1.to(device).to(torch.float32)
        fr2 = fr2.to(device).to(torch.float32)
        trg = trg.to(device).to(torch.float32)
        logits = model(fr1, fr2)
        loss = loss_fun(logits, trg) 
        t_loss += loss.to("cpu").item()
            # del x, y
        torch.cuda.empty_cache()
    return t_loss/len(test)


# %%
model, t_losses = learning(model, train_dl, optimizer, loss_fun, device, epochs)

# %%
plt.plot(t_losses)

# %% [markdown]
# As we can see it hasnt overfeeted yet thus with more training time we can get much better result than 2 on test data

# %%
evluate(model, test_dl, loss_fun, device)

# %%
@torch.no_grad()
def predict(model, fr1, fr2, device):
    model.eval()
    fr_ = fr1.to(device).to(torch.float32).unsqueeze(0)
    fr__ = fr2.to(device).to(torch.float32).unsqueeze(0)
    return model(fr_,fr__).to("cpu").item()

# %%
fr,fr_,trg = test_ds[100]
predict(model, fr, fr_, device), trg

# %%
plt.imshow(fr_.permute((1,2,0)).to("cpu").numpy())

# %%
def add_numbers_to_image(img, num1, num2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    blue = (255, 0, 0)
    green = (0, 255, 0)
    text = str(num1)
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, 2)[0]
    x = 10
    y = 20
    cv2.putText(img, text, (x, y), font, font_scale, blue, 2)
    text = str(num2)
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, 2)[0]
    x = 10
    y += text_height + 10
    cv2.putText(img, text, (x, y), font, font_scale, green, 2)
    return img

def create_video_from_images(dataset, fps, output_file):
    img,_,_ = dataset[0]
    height, width, layers = img.permute((1,2,0)).to("cpu").numpy().shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for i in tqdm(range(len(dataset))):
        fr,fr_,trg = dataset[i]
        img = add_numbers_to_image(fr_.permute((1,2,0)).to("cpu").numpy(), round(predict(model, fr, fr_, device),2), round(trg.item(), 2))        
        video.write(img)
    video.release()

# %%
create_video_from_images(test_ds, 60, "/kaggle/working/Speed_test.mp4")

# %%
print("YOOOO")

# %%



