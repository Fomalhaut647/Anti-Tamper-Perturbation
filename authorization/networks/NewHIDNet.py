import torch
import torch.nn as nn
import numpy as np
from networks.mapper import MappingNetwork
from networks.encoder_decoder import *
from networks.discriminator import Discriminator
from networks.utils import VGGLoss
import yaml
import os


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class NewHIDNet:
    def __init__(
        self, device, gamma=0.5, input_mask=None, randomFlag=False, pixel_space=False
    ) -> None:
        super().__init__()
        self.device = device
        self.mapping_network = nn.Identity()
        config_path = os.path.abspath(f"../configs/authorization.yaml")
        self.config = load_config(config_path)
        self.encoder_decoder = EncoderDecoder(
            encode_message_length=self.config["message_length"],
            decode_message_length=self.config["message_length"],
            pixel_space=pixel_space,
            block_size=self.config["block_size"],
        ).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.randomFlag = randomFlag
        self.pixel_space = pixel_space

        # 创建频率权重图
        self.frequency_weight_map = self._create_frequency_weight_map(
            self.config["block_size"], 
            (512, 512)  # 假设图像尺寸为512x512
        ).to(self.device)

        if not randomFlag:
            self.random_mask = torch.bernoulli(torch.ones(3, 512, 512) * gamma).to(
                self.device
            )
            self.gamma = gamma
            if input_mask is not None:
                self.random_mask = torch.load(input_mask).to(self.device)
            # print(self.random_mask.sum()/self.random_mask.reshape(-1).shape[0])
        else:
            self.gamma = gamma
        self.init_optimizer()
        self.init_losses()

    def _create_frequency_weight_map(self, block_size, image_size):
        """
        创建频率权重图，低频部分权重高，高频部分权重低
        在DCT变换中，低频分量位于左上角(0,0)，高频分量位于右下角
        """
        h, w = image_size
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        # 为单个DCT块创建权重矩阵
        block_weight = torch.zeros(block_size, block_size)
        
        # 计算每个频率分量到左上角(0,0)的距离
        for i in range(block_size):
            for j in range(block_size):
                # 计算到左上角的欧氏距离（归一化到[0,1]）
                # 距离公式：sqrt(i^2 + j^2)
                distance = (i**2 + j**2) ** 0.5
                # 最大距离（右下角到左上角的距离）
                max_distance = ((block_size-1)**2 + (block_size-1)**2) ** 0.5
                # 归一化距离
                normalized_distance = distance / max_distance
                # 使用指数衰减函数赋予权重：低频（距离小）权重高，高频（距离大）权重低
                # 调整衰减系数可以控制权重衰减速度
                weight = torch.exp(-normalized_distance * 3.0)
                block_weight[i, j] = weight
        
        # 将块权重扩展到整个图像
        weight_map = block_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, block_size, block_size]
        weight_map = weight_map.repeat(1, 1, blocks_h, blocks_w)  # [1, 1, blocks_h*block_size, blocks_w*block_size]
        
        # 如果尺寸不匹配，进行裁剪（通常应该匹配）
        weight_map = weight_map[:, :, :h, :w]
        
        return weight_map

    
    def _weighted_mse_loss(self, input, target, weight_map):
        """
        带权重的MSE损失计算
        """
        # 扩展权重图以匹配输入的形状
        if weight_map.shape[0] != input.shape[0]:
            weight_map = weight_map.repeat(input.shape[0], 1, 1, 1)
        if weight_map.shape[1] != input.shape[1]:
            weight_map = weight_map.repeat(1, input.shape[1], 1, 1)
        
        # 计算加权MSE
        return self.mse_loss(input * weight_map, target * weight_map)

    def init_optimizer(self):
        self.optimizer_enc_dec = torch.optim.Adam(
            list(self.encoder_decoder.parameters())
            + list(self.mapping_network.parameters()),
            lr=self.config["train"]["lr"],
        )
        self.optimizer_discrim = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config["train"]["lr"]
        )

    def init_losses(self):
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.vgg_loss = None
        self.adversarial_loss = self.config["train"]["adversarial_loss"]
        self.encoder_loss = self.config["train"]["encoder_loss"]
        self.decoder_loss = self.config["train"]["decoder_loss"]
        self.reg_loss = self.config["train"]["reg_loss"]

    def train_one_batch(self, batch):

        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), 1.0, device=self.device)
            d_target_label_encoded = torch.full(
                (batch_size, 1), 0.0, device=self.device
            )
            g_target_label_encoded = torch.full(
                (batch_size, 1), 1.0, device=self.device
            )

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(
                d_on_cover, d_target_label_cover
            )
            d_loss_on_cover.backward()

            # train on fake
            if not self.randomFlag:
                (
                    encoded_images,
                    decoded_messages,
                    random_mask,
                    encoded_dct,
                    img_dct_masked,
                ) = self.encoder_decoder(
                    images, self.mapping_network(messages), random_mask=self.random_mask
                )
            else:
                random_mask = torch.bernoulli(
                    torch.ones(images.shape[0], 3, 512, 512) * self.gamma
                ).to(self.device)
                (
                    encoded_images,
                    decoded_messages,
                    random_mask,
                    encoded_dct,
                    img_dct_masked,
                ) = self.encoder_decoder(
                    images, self.mapping_network(messages), random_mask=random_mask
                )
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(
                d_on_encoded, d_target_label_encoded
            )

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(
                d_on_encoded_for_enc, g_target_label_encoded
            )

            # if self.vgg_loss == None:
            #     g_loss_enc = self.mse_loss(encoded_images, images)
            # else:
            #     vgg_on_cov = self.vgg_loss(images)
            #     vgg_on_enc = self.vgg_loss(encoded_images)
            #     g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc) + self.mse_loss(
            #         encoded_images, images
            #     )
            # 修改点：将g_loss_enc替换为g_loss_overflow
            # 只惩罚超出[-1, 1]范围的像素值，不强制逼近原图
            overflow_positive = torch.nn.functional.relu(encoded_images - 1)  # 大于1的部分
            overflow_negative = torch.nn.functional.relu(-1 - encoded_images)  # 小于-1的部分
            g_loss_overflow = (overflow_positive + overflow_negative).mean()

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # g_loss_reg = self.mse_loss(encoded_dct, img_dct_masked)
            # 修改点：使用加权的MSE计算g_loss_reg
            g_loss_hvs = self._weighted_mse_loss(encoded_dct, img_dct_masked, self.frequency_weight_map)
            if not self.pixel_space:
                reg_co = self.reg_loss
            else:
                reg_co = 0
            # g_loss = (
            #     self.adversarial_loss * g_loss_adv
            #     + self.encoder_loss * g_loss_enc
            #     + self.decoder_loss * (g_loss_dec + reg_co * g_loss_reg)
            # )
            g_loss = (
                self.adversarial_loss * g_loss_adv
                + self.encoder_loss * g_loss_overflow
                + self.decoder_loss * g_loss_dec
                + reg_co * g_loss_hvs
            )

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(
            np.abs(decoded_rounded - messages.detach().cpu().numpy())
        ) / (batch_size * messages.shape[1])

        losses = {
            "loss           ": g_loss.item(),
            # "encoder_mse    ": g_loss_enc.item(), # L_rec
            "encoder_mse    ": g_loss_overflow.item(), # L_overflow
            "dec_mse        ": g_loss_dec.item(), # L_con
            # "dec_reg        ": g_loss_reg.item(), # L_reg
            "dec_reg        ": g_loss_hvs.item(), # L_hvs
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(), # L_adv
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
        }
        return losses, (encoded_images, random_mask, decoded_messages)

    def val_one_batch(self, batch):

        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():

            d_target_label_cover = torch.full((batch_size, 1), 1.0, device=self.device)
            d_target_label_encoded = torch.full(
                (batch_size, 1), 0.0, device=self.device
            )
            g_target_label_encoded = torch.full(
                (batch_size, 1), 1.0, device=self.device
            )

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(
                d_on_cover, d_target_label_cover
            )

            if not self.randomFlag:
                (
                    encoded_images,
                    decoded_messages,
                    random_mask,
                    encoded_dct,
                    img_dct_masked,
                ) = self.encoder_decoder(
                    images, self.mapping_network(messages), random_mask=self.random_mask
                )
            else:
                random_mask = torch.bernoulli(
                    torch.ones(images.shape[0], 3, 512, 512) * self.gamma
                ).to(self.device)
                (
                    encoded_images,
                    decoded_messages,
                    random_mask,
                    encoded_dct,
                    img_dct_masked,
                ) = self.encoder_decoder(
                    images, self.mapping_network(messages), random_mask=random_mask
                )

            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(
                d_on_encoded, d_target_label_encoded
            )

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(
                d_on_encoded_for_enc, g_target_label_encoded
            )

            # if self.vgg_loss == None:
            #     g_loss_enc = self.mse_loss(encoded_images, images)
            # else:
            #     vgg_on_cov = self.vgg_loss(images)
            #     vgg_on_enc = self.vgg_loss(encoded_images)
            #     g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc) + self.mse_loss(
            #         encoded_images, images
            #     )
            # 修改点：将g_loss_enc替换为g_loss_overflow
            # 只惩罚超出[-1, 1]范围的像素值，不强制逼近原图
            overflow_positive = torch.nn.functional.relu(encoded_images - 1)  # 大于1的部分
            overflow_negative = torch.nn.functional.relu(-1 - encoded_images)  # 小于-1的部分
            g_loss_overflow = (overflow_positive + overflow_negative).mean()

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # g_loss_reg = self.mse_loss(encoded_dct, img_dct_masked)
            # 修改点：使用加权的MSE计算g_loss_reg
            g_loss_hvs = self._weighted_mse_loss(encoded_dct, img_dct_masked, self.frequency_weight_map)
            if not self.pixel_space:
                reg_co = self.reg_loss
            else:
                reg_co = 0
            # g_loss = (
            #     self.adversarial_loss * g_loss_adv
            #     + self.encoder_loss * g_loss_enc
            #     + self.decoder_loss * (g_loss_dec + reg_co * g_loss_reg)
            # )
            g_loss = (
                self.adversarial_loss * g_loss_adv
                + self.encoder_loss * g_loss_overflow
                + self.decoder_loss * g_loss_dec
                + reg_co * g_loss_hvs
            )
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)

        bitwise_avg_err = np.sum(
            np.abs(decoded_rounded - messages.detach().cpu().numpy())
        ) / (batch_size * messages.shape[1])

        losses = {
            "loss           ": g_loss.item(),
            # "encoder_mse    ": g_loss_enc.item(), # L_rec
            "encoder_mse    ": g_loss_overflow.item(), # L_overflow
            "dec_mse        ": g_loss_dec.item(), # L_con
            # "dec_reg        ": g_loss_reg.item(), # L_reg
            "dec_reg        ": g_loss_hvs.item(), # L_hvs
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(), # L_adv
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
        }
        return losses, (encoded_images, random_mask, decoded_messages)