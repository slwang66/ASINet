import os
import numpy as np
import scipy.io as scio
import scipy.sparse
import torch
import spconv.pytorch as spconv

from model.model import rec, Dilation, OACNNs


device = torch.device("cuda:0")

net1 = rec().to(device)
net2 = Dilation().to(device)
net3 = OACNNs(1, 1).to(device)


def load(net, path):
    ckpt = torch.load(path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    net.load_state_dict({**net.state_dict(), **ckpt})


load(net1, "./weight/denoise.pth")
load(net3, "./weight/rec.pth")

net1.eval()
net2.eval()
net3.eval()


data_dir = "./Data"
save_dir = "./result"
os.makedirs(save_dir, exist_ok=True)


def recon(path):

    data = scio.loadmat(path)
    h, w = data["depth"].shape

    spad = data["spad"]
    if scipy.sparse.issparse(spad):
        spad = spad.toarray()

    spad = np.asarray(spad, np.float32)
    spad = torch.from_numpy(
        np.transpose(
            spad.reshape(1, 1, h, w, -1),
            (0, 4, 3, 2, 1)
        )
    ).to(device)

    with torch.no_grad():

        out, _ = net1(spad)

        mask = net2(out).permute(0, 2, 3, 4, 1) > 0

        spad = spconv.SparseConvTensor.from_dense(
            (spad + 1) * mask
        )

        out = torch.squeeze(net3(spad))
        out = torch.softmax(out, 0)

        depth = (
            out *
            torch.arange(
                out.shape[0],
                device=device
            ).view(-1, 1, 1)
        ).sum(0)

    depth = depth.cpu().numpy() * 3e8 * 80e-12 / 2 + 0.012

    scio.savemat(
        os.path.join(
            save_dir,
            os.path.basename(path).replace(".mat", "_rec.mat")
        ),
        {"depth": depth}
    )


for f in os.listdir(data_dir):
    if f.endswith(".mat"):
        recon(os.path.join(data_dir, f))