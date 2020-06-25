from architectures.ArchitecturesUtils import *
from architectures.Generators import *
from architectures.Discriminators import *
from architectures.Losses import *
from architectures.Evaluations import *
import datetime
import numpy as np
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import json
from interpret_segmentation import hdm

class TrainerCycleGAN():
    def __init__(self, input_shape, opt):
        self.c, self.h, self.w = input_shape
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = opt.n_epochs
        self.batch_size = opt.batch_size
        self.model_name = opt.model_name

        patch_size = int(self.h // 2 ** 4)
        self.discriminator_patch = (1, patch_size, patch_size)
        self.lambda_cyc = opt.lambda_cyc
        self.lambda_identity = opt.lambda_id
        self.lambda_seg = opt.lambda_seg
        self.lambda_adv = opt.lambda_gan

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()


        self.Di = Dis(self.c, self.c).to(self.device)
        self.Ds = Dis(self.c, self.c).to(self.device)

        # self.optimizer_d_i = torch.optim.RMSprop(self.Di.parameters(), lr=opt.lr, weight_decay=1e-7, momentum=0.9)
        # self.optimizer_d_s = torch.optim.RMSprop(self.Ds.parameters(), lr=opt.lr, weight_decay=1e-7, momentum=0.9)
        self.optimizer_d_i = torch.optim.Adam(self.Di.parameters(), lr=opt.lr, weight_decay=1e-7, betas=(opt.b1, opt.b2))
        self.optimizer_d_s = torch.optim.Adam(self.Ds.parameters(), lr=opt.lr, weight_decay=1e-7, betas=(opt.b1, opt.b2))
        self.scheduler_d_i = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d_i, mode='max', factor=0.5)
        self.scheduler_d_s = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d_s, mode='max', factor=0.5)

        self.Gis = UnetGen(self.c, self.c).to(self.device)
        self.Gsi = UnetGen(self.c, self.c).to(self.device)

        # self.optimizer_g_is = torch.optim.RMSprop(self.Gis.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
        # self.optimizer_g_si = torch.optim.RMSprop(self.Gis.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
        self.optimizer_g_is = torch.optim.Adam(self.Gis.parameters(), lr=opt.lr, weight_decay=1e-7, betas=(opt.b1, opt.b2))
        self.optimizer_g_si = torch.optim.Adam(self.Gis.parameters(), lr=opt.lr, weight_decay=1e-7, betas=(opt.b1, opt.b2))
        self.scheduler_g_is = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g_is, mode='max', factor=0.5)
        self.scheduler_g_si = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g_si, mode='max', factor=0.5)

        self.logs = {'epochs': [], 'd_loss': [], 'g_loss': [], 'mIoU': [], 'mDice': []}
        self.model_dir = os.path.join("ExportedModels")
        return

    def train(self, data_loader_t, data_loader_v, sample_interval=10, data_name='lung'):

        best_iou = 0
        best_dice = 0

        # Adversarial loss ground truths
        valid = Variable(self.tensor(np.ones((self.batch_size, *self.discriminator_patch))),
                         requires_grad=False).to(self.device)
        fake = Variable(self.tensor(np.zeros((self.batch_size, *self.discriminator_patch))),
                        requires_grad=False).to(self.device)

        for epoch in range(self.epochs):
            d_loss_list = []
            g_loss_list = []
            self.Gis.train()
            self.Gsi.train()
            self.Di.train()
            self.Ds.train()
            for batch_i, (imgs, segs) in enumerate(data_loader_t):
                start_time = datetime.datetime.now()

                imgs = imgs.to(self.device)
                segs = segs.to(self.device)

                # ----------------------
                #  Train Discriminators
                # ----------------------
                self.optimizer_d_i.zero_grad()
                self.optimizer_d_s.zero_grad()

                fake_segs = self.Gis(imgs)
                fake_imgs = self.Gsi(segs)

                d_imgs_real = self.Di(imgs)
                d_imgs_fake = self.Di(fake_imgs)
                d_img_loss = self.mse(d_imgs_real, valid) + self.mse(d_imgs_fake, fake)
                d_img_loss.backward()
                self.optimizer_d_i.step()

                d_segs_real = self.Ds(segs)
                d_segs_fake = self.Ds(fake_segs)
                d_segs_loss = self.mse(d_segs_real, valid) + self.mse(d_segs_fake, fake)
                d_segs_loss.backward()
                nn.utils.clip_grad_value_(self.Di.parameters(), 0.1)
                nn.utils.clip_grad_value_(self.Ds.parameters(), 0.1)
                self.optimizer_d_s.step()

                d_loss = d_segs_loss + d_img_loss
                d_loss_list.append(d_loss.item())

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_g_is.zero_grad()
                self.optimizer_g_si.zero_grad()

                fake_segs = self.Gis(imgs)
                fake_imgs = self.Gsi(segs)

                regenerated_imgs = self.Gsi(fake_segs)
                regenerated_segs = self.Gsi(fake_imgs)

                identity_imgs = self.Gsi(imgs)
                identity_segs = self.Gis(segs)

                identity_loss = self.lambda_identity * (self.mae(identity_segs, segs) +
                                                        self.mae(identity_imgs, imgs))
                cycle_loss = self.lambda_cyc * (self.mae(regenerated_segs, segs) +
                                                self.mae(regenerated_imgs, imgs))
                adverserial_loss = self.lambda_adv * (self.mse(self.Di(fake_imgs), valid) +
                                                      self.mse(self.Ds(fake_segs), valid))
                segmentation_loss = self.lambda_seg * (self.bce(fake_segs, segs) +
                                                       self.bce(regenerated_segs, segs))
                g_loss = cycle_loss + adverserial_loss + segmentation_loss + identity_loss

                g_loss.backward()
                nn.utils.clip_grad_value_(self.Gis.parameters(), 0.1)
                nn.utils.clip_grad_value_(self.Gsi.parameters(), 0.1)

                self.optimizer_g_is.step()
                self.optimizer_g_si.step()

                g_loss_list.append(g_loss.item())

                elapsed_time = datetime.datetime.now() - start_time

                # print the progress
                print(
                    "[ Epoch %d/%d ] [ Batch %d/%d ] [ D loss: %06f ] [ G loss: %06f, adv: %06f, recon: %06f, id: %06f, seg: %06f ] [ time: %s ]" \
                    % (epoch, self.epochs,
                       batch_i, len(data_loader_t),
                       d_loss.item(),
                       g_loss.item(),
                       adverserial_loss.item(),
                       cycle_loss.item(),
                       identity_loss.item(),
                       segmentation_loss.item(),
                       elapsed_time))


            if epoch % sample_interval == 0:
                self.sampleImages(epoch, batch_i, data_loader_v, data_name)
            m_iou, m_dice = self.evaluate(data_loader_v)

            self.scheduler_d_i.step(m_dice)
            self.scheduler_d_s.step(m_dice)
            self.scheduler_g_is.step(m_dice)
            self.scheduler_g_si.step(m_dice)

            if m_iou > best_iou and m_dice > best_dice:
                self.saveModels(data_name)
                best_dice = m_dice
                best_iou = m_iou

            # save epochs statistics
            self.logs['epochs'].append(epoch)
            self.logs['d_loss'].append(d_loss_list)
            self.logs['g_loss'].append(g_loss_list)
            self.logs['mIoU'].append(m_iou)
            self.logs['mDice'].append(m_dice)

        # self.saveModels(data_name)
        json.dump(self.logs, open("{:}_{:}_train_stat.json".format(data_name, self.epochs), "w"))
        return

    def sampleImages(self, epoch, batch_i, data_loader, data_name):

        self.Gis.eval()
        self.Gsi.eval()

        r, c = 2, 3
        for img_idx, (imgs, segs) in enumerate(data_loader):
            imgs = imgs.to(self.device)
            segs = segs.to(self.device)

            os.makedirs('images/%s' % data_name, exist_ok=True)

            fake_imgs = self.Gsi(segs)
            fake_segs = self.Gis(imgs)
            fake_segs = torch.sigmoid(fake_segs)

            regenerated_imgs = self.Gsi(fake_segs)
            regenerated_segs = self.Gsi(fake_imgs)
            regenerated_segs = torch.sigmoid(regenerated_segs)

            gen_imgs = [imgs, fake_segs, regenerated_imgs, segs, fake_imgs, regenerated_segs]

            titles = ['Original', 'Translated', 'Reconstructed']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    if j == 2:
                        im = gen_imgs[cnt]
                        im = im.squeeze().cpu().detach().numpy()
                        im = im > 0.5
                        im = im.astype(np.uint8)
                    else:
                        im = 0.5 * gen_imgs[cnt].squeeze().cpu().detach().numpy() + 0.5
                    axs[i, j].imshow(im)
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/%s/%d_%d_%d.png" % (data_name, img_idx, epoch, batch_i))
            plt.close()

    def evaluate(self, loader):
        self.Gis.eval()
        ious = np.zeros(len(loader))
        dices = np.zeros(len(loader))
        print(15 * "~-")
        for i, (im, seg) in enumerate(loader):
            im = im.to(self.device)
            seg = seg

            pred = self.Gis(im)
            pred = torch.sigmoid(pred)
            pred = pred.cpu()
            pred = pred.detach().numpy()[0, 0, :, :]
            mask = seg.numpy()[0, 0, :, :]

            # Binarize masks
            gt = mask > 0.5
            pr = pred > 0.5

            ious[i] = IoU(gt, pr)
            dices[i] = Dice(gt, pr)
            print("results: IoU: {:f}, DICE: {:f}".format(ious[i], dices[i]))

        print('Mean IoU:', ious.mean())
        print('Mean Dice:', dices.mean())
        print(15 * "~-")
        return ious.mean(), dices.mean()

    def saveModels(self, data_name):
        print("Saving Model!")

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        g_path = os.path.join(self.model_dir, "Generators")
        if not os.path.exists(g_path):
            os.mkdir(g_path)

        torch.save(self.Gis.state_dict(), os.path.join(g_path, "{:}_{:}_Gis_E_{:}.pt".format(data_name,
                                                                                             self.model_name,
                                                                                             self.epochs)))
        torch.save(self.Gsi.state_dict(), os.path.join(g_path, "{:}_{:}_Gsi_E_{:}.pt".format(data_name,
                                                                                             self.model_name,
                                                                                             self.epochs)))

        d_path = os.path.join(self.model_dir, "Discriminators")
        if not os.path.exists(d_path):
            os.mkdir(d_path)

        torch.save(self.Ds.state_dict(), os.path.join(d_path, "{:}_{:}_Ds_E_{:}.pt".format(data_name,
                                                                                           self.model_name,
                                                                                           self.epochs)))
        torch.save(self.Di.state_dict(), os.path.join(d_path, "{:}_{:}_Di_E_{:}.pt".format(data_name,
                                                                                           self.model_name,
                                                                                           self.epochs)))
        return


class TesterCycleGAN():
    def __init__(self, input_shape, opt):
        self.c, self.h, self.w = input_shape
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = opt.n_epochs
        self.batch_size = opt.batch_size

        self.Gis = UnetGen(self.c, self.c).to(self.device)

    def test(self, data_loader, data_name):
        g_path = os.path.join("ExportedModels", "Generators",
                              "{:}_cycle_Gis_E_{:}.pt".format(data_name, self.epochs))
        if self.device == 'cuda':
            self.Gis.load_state_dict(torch.load(g_path))
        else:
            self.Gis.load_state_dict(torch.load(g_path, map_location = 'cpu'))

        self.Gis.eval()
        ious = np.zeros(len(data_loader))
        dices = np.zeros(len(data_loader))
        hds = np.zeros(len(data_loader))
        HD = hdm.HausdorffDistanceMasks(256, 256)
        print(15 * "~-")
        for i, (im, seg) in enumerate(data_loader):
            im = im.to(self.device)
            seg = seg

            pred = self.Gis(im)
            pred = torch.sigmoid(pred)
            pred = pred.cpu()
            pred = pred.detach().numpy()[0, 0, :, :]
            mask = seg.detach().numpy()[0, 0, :, :]
            # Binarize masks
            gt = mask > 0.5
            pr = pred > 0.5

            ious[i] = IoU(gt, pr)
            dices[i] = Dice(gt, pr)
            hds[i] = HD.calculate_distance(pr, gt)
            print("results: IoU: {:f}, DICE: {:f}, HD: {:f}".format(ious[i], dices[i], hds[i]))

        print('Mean IoU:', ious.mean())
        print('Mean Dice:', dices.mean())
        print('Mean HD:', hds.mean())
        print(15 * "~-")
        return
