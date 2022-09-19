from __future__ import absolute_import, division, print_function
from datetime import datetime
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import torchvision
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks

def build_extractor_depth_encoder_decoder(pretrained_path):
    extractor = networks.test_hr_encoder.hrnet18(False)
    decoder = networks.HRDepthDecoder([ 64, 18, 36, 72, 144 ], [0, 1, 2, 3])
    model_dict_e = extractor.state_dict()
    model_dict_d = decoder.state_dict()
    if pretrained_path:
        e_dict = torch.load(os.path.join(pretrained_path, "encoder.pth"))
        extractor.load_state_dict({k: v for k, v in e_dict.items() if k in model_dict_e})
        for param in extractor.parameters():
            param.requires_grad = False
        d_dict = torch.load(os.path.join(pretrained_path, "depth.pth"))
        decoder.load_state_dict({k: v for k, v in d_dict.items() if k in model_dict_d})
        for param in decoder.parameters():
            param.requires_grad = False
    return extractor, decoder

def build_extractor_depth_encoder_decoder_fusion(pretrained_path):
    fusion = networks.Fusion()
    extractor = networks.test_hr_encoder.hrnet18(False)
    decoder = networks.HRDepthDecoder([ 64, 18, 36, 72, 144 ], [0, 1, 2, 3])
    model_dict_e = extractor.state_dict()
    model_dict_d = decoder.state_dict()
    model_dict_f = fusion.state_dict()
    if pretrained_path:
        e_dict = torch.load(os.path.join(pretrained_path, "encoder.pth"))
        extractor.load_state_dict({k: v for k, v in e_dict.items() if k in model_dict_e})
        for param in extractor.parameters():
            param.requires_grad = False
        d_dict = torch.load(os.path.join(pretrained_path, "depth.pth"))
        decoder.load_state_dict({k: v for k, v in d_dict.items() if k in model_dict_d})
        for param in decoder.parameters():
            param.requires_grad = False
        f_dict = torch.load(os.path.join(pretrained_path, "fusion.pth"))
        fusion.load_state_dict({k: v for k, v in f_dict.items() if k in model_dict_f})
        for param in fusion.parameters():
            param.requires_grad = False
    return fusion, extractor, decoder
def build_extractor(num_layers, pretrained_path):
    extractor = networks.ResnetEncoder(50, None)
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        for name, param in extractor.state_dict().items():
            extractor.state_dict()[name].copy_(checkpoint['state_dict']['Encoder.' + name])
        for param in extractor.parameters():
            param.requires_grad = False
    return extractor

class Trainer:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name, current_time_date) 

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'

        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        #able if not using use_stereo or frame_ids !=0
        #use_stereo defualt disable
        #frame_ids defualt =[0,-1,1]

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.models["encoder"] = networks.test_hr_encoder.hrnet18(True)
        self.models["encoder"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
        #self.models["fusion"] = networks.Fusion()
        if self.opt.auto_prtrained_model:
            self.extractor = build_extractor(50, self.opt.auto_prtrained_model)
            self.extractor.to(self.device)
        else:
            self.extractor = False

        # if self.opt.depth_encoder_decoder_fusion:
        #     self.d_fusion, self.d_encoder, self.d_decoder = build_extractor_depth_encoder_decoder_fusion(self.opt.depth_encoder_decoder_fusion)
        #     self.d_fusion.to(self.device)
        #     self.d_encoder.to(self.device)
        #     self.d_decoder.to(self.device)
        if self.opt.depth_encoder_decoder:
            self.d_encoder_s, self.d_decoder_s = build_extractor_depth_encoder_decoder(self.opt.depth_encoder_decoder)
            self.d_encoder_s.to(self.device)
            self.d_decoder_s.to(self.device)
        # if self.load_depth_encoder:
        #     self.depth_encoder = build_extractor_depth_encoder(self.load_depth_encoder)
        para_sum = sum(p.numel() for p in self.models['encoder'].parameters())
        print('params in encoder',para_sum)
        
        self.models["depth"] = networks.HRDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        
        
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        #self.models["fusion"].to(self.device)
        #self.parameters_to_train += list(self.models["fusion"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        para_sum = sum(p.numel() for p in self.models['depth'].parameters())
        print('params in depth decdoer',para_sum)

        if self.use_pose_net:#use_pose_net = True
            if self.opt.pose_model_type == "separate_resnet":#defualt=separate_resnet  choice = ['normal or shared']
                
                # 输出 translation 和 rotation的PoseNet
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)#num_input_images=2
                
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                
                # 只输出 translation PoseNet
                '''
                self.models["pose_encoder_only_t"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)#num_input_images=2
                '''
                self.models["pose_only_t"] = networks.PoseDecoder_only_t(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

                # 只输出 rotation PoseNet
                '''
                self.models["pose_encoder_only_r"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)#num_input_images=2
                '''

                self.models["pose_only_r"] = networks.PoseDecoder_only_r(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose_encoder"].cuda()
            self.models["pose"].cuda()
            #self.models["pose_encoder_only_t"].cuda()
            self.models["pose_only_t"].cuda()
            #self.models["pose_encoder_only_r"].cuda()
            self.models["pose_only_r"].cuda()

            self.parameters_to_train += list(self.models["pose"].parameters())
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)#learning_rate=1e-4
        #self.model_optimizer = optim.Adam(self.parameters_to_train, 0.5 * self.opt.learning_rate)#learning_rate=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset
                         }
        self.dataset_k = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        #change trainset
        train_filenames_k = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames_k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        #dataloader for kitti
        train_dataset_k = self.dataset_k(
            self.opt.data_path, train_filenames_k, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext='.jpg')
        self.train_loader_k = DataLoader(
            train_dataset_k, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        #val_dataset = self.dataset(
        val_dataset = self.dataset_k( 
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch_k = train_dataset_k.__len__() // self.opt.batch_size

        self.backproject_depth = {}
        self.project_3d = {}

        # feature generated
        self.backproject_feature = Backproject(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))
        self.project_feature = Project(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))
        self.backproject_feature.to(self.device)
        self.project_feature.to(self.device)

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset_k), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.init_time = time.time()
        if isinstance(self.opt.load_weights_folder,str):
            self.epoch_start = int(self.opt.load_weights_folder[-1]) + 1
        else:
            self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Threads: " + str(torch.get_num_threads()))
        print("Training")
        self.set_train()
        self.every_epoch_start_time = time.time()
        
        for batch_idx, inputs in enumerate(self.train_loader_k):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000#log_fre 's defualt = 250
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                #self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        
        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))
   
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            # f_features = self.models["encoder"](inputs[("color_aug", -1, 0)])
            # t_features = self.models["encoder"](inputs[("color_aug", 1, 0)])
            # # f_features = self.depth_encoder(inputs[("color_aug", -1, 0)])
            # # t_features = self.depth_encoder(inputs[("color_aug", 1, 0)])
            # f = [f_features] + [features] + [t_features]
            # features = self.models["fusion"](f)
            # f_depth = self.d_decoder(self.d_encoder(inputs[("color", -1, 0)]))
            # s_depth = self.d_decoder(self.d_encoder(inputs[("color", 0, 0)]))
            # t_depth = self.d_decoder(self.d_encoder(inputs[("color", 1, 0)]))

            #sys_frame = self.models["fusion"](inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)], f_depth[("disp", 0)], s_depth[("disp", 0)], t_depth[("disp", 0)])
            features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            outputs = self.models["depth"](features)
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
            #different form 1:*:* depth maps ,it will output 2:*:* mask maps

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        self.generate_features_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #pose_feats is a dict:
            #key:
            """"keys
                0
                -1
                1
            """
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        # translation 和 rotation 均预测
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        '''
                        # 只预测 translation
                        pose_inputs_only_t = [self.models["pose_encoder_only_t"](torch.cat(pose_inputs, 1))]
                        # 只预测 rotation
                        pose_inputs_only_r = [self.models["pose_encoder_only_r"](torch.cat(pose_inputs, 1))]
                        '''
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    # translations 和 rotation 均预测的网路
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    # 单个的translations 和 rotation
                    axisangle_only_r = self.models["pose_only_r"](pose_inputs)
                    translation_only_t = self.models["pose_only_t"](pose_inputs)
                    outputs[("axisangle_only_r", 0, f_i)] = axisangle_only_r
                    outputs[("translation_only_t", 0, f_i)] = translation_only_t
                    #axisangle and translation are two 2*1*3 matrix
                    #f_i=-1,1
                    # Invert the matrix if the frame id is negative
                    # translation 和 rotation均预测的网络
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    axisangle_temp = torch.zeros_like(axisangle)
                    outputs[("cam_T_cam_only_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_temp[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam_r_and_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_only_r[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    translation_temp = torch.zeros_like(translation)
                    outputs[("cam_T_cam_only_r", 0, f_i)] = transformation_from_parameters(
                        axisangle_only_r[:, 0], translation_temp[:, 0], invert=(f_i < 0))
                    

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    # t r 均预测
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                    
                    # 只预测 translation
                    pose_inputs_only_t = [self.models["pose_encoder_only_t"](torch.cat(pose_inputs, 1))]
                    # 只预测 rotation
                    pose_inputs_only_r = [self.models["pose_encoder_only_r"](torh.cat(pose_inputs, 1))]
                    
                    

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)
            # 单个的translations 和 rotation
            axisangle_only_r = self.models["pose_only_r"](pose_inputs)
            translation_only_t = self.models["pose_only_t"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("axisangle_only_r", 0, f_i)] = axisangle_only_r
                    outputs[("translation_only_t", 0, f_i)] = translation_only_t

                    axisangle_temp = torch.zeros_like(axisangle)
                    outputs[("cam_T_cam_only_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_temp[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam_r_and_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_only_r[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs
 
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)#disp_to_depth function is in layers.py

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    # 同时预测r 和 t 的网络得到的结果
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    T_only_t = outputs[("cam_T_cam_only_t", 0, frame_id)]
                    T_r_and_t = outputs[("cam_T_cam_r_and_t", 0, frame_id)]
                    T_only_r = outputs[("cam_T_cam_only_r", 0, frame_id)]
                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # 同时预测r和t的网络
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                
                # 只预测t
                pix_coords_only_t = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T_only_t)
                outputs[("sample_only_t", frame_id, scale)] = pix_coords_only_t

                outputs[("color_only_t", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample_only_t", frame_id, scale)],
                    padding_mode="border")

                # 只预测r
                pix_coords_only_r = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T_only_r)
                outputs[("sample_only_r", frame_id, scale)] = pix_coords_only_r

                outputs[("color_only_r", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample_only_r", frame_id, scale)],
                    padding_mode="border")

                # 单独的r和t组合
                pix_coords_r_and_t = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T_r_and_t)
                outputs[("sample_r_and_t", frame_id, scale)] = pix_coords_r_and_t

                outputs[("color_r_and_t", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample_r_and_t", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    #doing this
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    def generate_features_pred(self, inputs, outputs):
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [int(self.opt.height/2), int(self.opt.width/2)], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
                T_only_t = outputs[("cam_T_cam_only_t", 0, frame_id)]
                T_r_and_t = outputs[("cam_T_cam_r_and_t", 0, frame_id)]
                T_only_r = outputs[("cam_T_cam_only_r", 0, frame_id)]

            K = inputs[("K", 0)].clone()
            K[:, 0, :] /= 2
            K[:, 1, :] /= 2

            inv_K = torch.zeros_like(K)
            for i in range(inv_K.shape[0]):
                inv_K[i, :, :] = torch.pinverse(K[i, :, :])

            #cam_points = self.backproject_feature(depth, inv_K)
            #pix_coords = self.project_feature(cam_points, K, T)  # [b,h,w,2]

            #img = inputs[("color", frame_id, 0)]
            #src_f = self.extractor(img)[0]  # 这个地方取了第一维的特征，所以后面的也需要拿第一维的特征
            #outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")
            outputs[("feature", frame_id, 0)] = self.extractor(outputs[("color", frame_id, 0)])[0]

            # only t
            #pix_coords_only_t = self.project_feature(cam_points, K, T_only_t)  # [b,h,w,2]
            outputs[("feature_only_t", frame_id, 0)] = self.extractor(outputs[("color_only_t", frame_id, 0)])[0]
            #outputs[("feature_only_t", frame_id, 0)] = F.grid_sample(src_f, pix_coords_only_t, padding_mode="border")

            # only r
            #pix_coords_only_r = self.project_feature(cam_points, K, T_only_r)  # [b,h,w,2]
            outputs[("feature_only_r", frame_id, 0)] = self.extractor(outputs[("color_only_r", frame_id, 0)])[0]
            #outputs[("feature_only_r", frame_id, 0)] = F.grid_sample(src_f, pix_coords_only_r, padding_mode="border")

            # t and r
            #pix_coords_r_and_t = self.project_feature(cam_points, K, T_r_and_t)  # [b,h,w,2]
            outputs[("feature_r_and_t", frame_id, 0)] = self.extractor(outputs[("color_r_and_t", frame_id, 0)])[0]
            #outputs[("feature_r_and_t", frame_id, 0)] = F.grid_sample(src_f, pix_coords_r_and_t, padding_mode="border")
            
    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        visit = False
        #f_depth = self.d_decoder_s(self.d_encoder_s(inputs[("color", -1, 0)]))
        s_depth = self.d_decoder_s(self.d_encoder_s(inputs[("color", 0, 0)]))
        #t_depth = self.d_decoder_s(self.d_encoder_s(inputs[("color", 1, 0)]))

        #sys_frame = self.models["fusion"](inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)], f_depth[("disp", 0)], s_depth[("disp", 0)], t_depth[("disp", 0)])
        #target_depth_feature_t = self.models["encoder"](sys_frame)
        #target_depth_t = self.models["depth"](target_depth_feature_t)
        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []
            perceptional_losses = []

            loss += torch.abs(outputs[("disp", scale)] - s_depth[("disp", scale)]).mean()
            #loss += torch.abs(target_depth_t[("disp", scale)] - s_depth[("disp", scale)]).mean() * 0.01
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            """
            minimum perceptional loss
            """
            if not visit:
                for frame_id in self.opt.frame_ids[1:]:
                    src_f = outputs[("feature", frame_id, 0)]
                    src_f_only_t = outputs[("feature_only_t", frame_id, 0)]
                    src_f_r_and_t = outputs[("feature_r_and_t", frame_id, 0)]
                    src_f_only_r = outputs[("feature_only_r", frame_id, 0)]
                    tgt_f = self.extractor(inputs[("color", 0, 0)])[0]

                    src_f_loss = self.compute_perceptional_loss(tgt_f, src_f)
                    src_f_only_t_loss = self.compute_perceptional_loss(tgt_f, src_f_only_t)
                    src_f_r_and_t_loss = self.compute_perceptional_loss(tgt_f, src_f_r_and_t)
                    src_f_only_r_loss = self.compute_perceptional_loss(tgt_f, src_f_only_r)

                    final, _ = torch.min(torch.cat((src_f_loss, src_f_only_t_loss, src_f_r_and_t_loss, src_f_only_r_loss), 1), 1, True)
                    perceptional_losses.append(final)
                perceptional_loss = torch.cat(perceptional_losses, 1)

                min_perceptional_loss, _ = torch.min(perceptional_loss, dim=1)
                loss += 4 * self.opt.perception_weight * min_perceptional_loss.mean()
                visit = True


            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                pred_only_t = outputs[("color_only_t", frame_id, scale)]
                pred_r_and_t = outputs[("color_r_and_t", frame_id, scale)]
                pred_only_r = outputs[("color_only_r", frame_id, scale)] 

                loss_pred = self.compute_reprojection_loss(pred, target)
                loss_pred_only_t = self.compute_reprojection_loss(pred_only_t, target)
                loss_pred_r_and_t = self.compute_reprojection_loss(pred_r_and_t, target)
                loss_pred_only_r = self.compute_reprojection_loss(pred_only_r, target)
                final, _ = torch.min(torch.cat((loss_pred, loss_pred_only_t, loss_pred_r_and_t, loss_pred_only_r), 1), 1, True)
                reprojection_losses.append(final)
                # l_pred_split = torch.split(loss_pred, 1, dim=0)
                # l_pred_only_t_split = torch.split(loss_pred_only_t, 1, dim=0)
                # l_pred_r_and_t_split = torch.split(loss_pred_r_and_t, 1, dim=0)

                # temp_reprojection_losses_for_one_image = []
                
                # for i in range(len(l_pred_split)):
                #     a, b, c = torch.sum(l_pred_split[i]), torch.sum(l_pred_only_t_split[i]), torch.sum(l_pred_r_and_t_split[i])
                #     if b <= a and b <= c:
                #         temp_reprojection_losses_for_one_image.append(l_pred_only_t_split[i])
                #         times[0] += 1
                #         pose_loss[0] += torch.mean(torch.abs(outputs[("translation", 0, frame_id)][i, :, :, :] - outputs[("translation_only_t", 0, frame_id)][i, :, :, :]))
                #     elif a <= b and a <= c:
                #         temp_reprojection_losses_for_one_image.append(l_pred_split[i])
                #         times[0] += 1
                #         times[1] += 1
                #         pose_loss[0] += torch.mean(torch.abs(outputs[("translation", 0, frame_id)][i, :, :, :] - outputs[("translation_only_t", 0, frame_id)][i, :, :, :]))
                #         pose_loss[1] += torch.mean(torch.abs(outputs[("axisangle", 0, frame_id)][i, :, :, :] - outputs[("axisangle_only_r", 0, frame_id)][i, :, :, :]))
                #     else:
                #         temp_reprojection_losses_for_one_image.append(l_pred_r_and_t_split[i])
                #         times[0] += 1
                #         times[1] += 1
                #         pose_loss[0] += torch.mean(torch.abs(outputs[("translation", 0, frame_id)][i, :, :, :] - outputs[("translation_only_t", 0, frame_id)][i, :, :, :]))
                #         pose_loss[1] += torch.mean(torch.abs(outputs[("axisangle", 0, frame_id)][i, :, :, :] - outputs[("axisangle_only_r", 0, frame_id)][i, :, :, :]))
                # temp_reprojection_losses_for_one_image = torch.cat(temp_reprojection_losses_for_one_image, 0)
                # reprojection_losses.append(temp_reprojection_losses_for_one_image)
                # 此处写pose的idea的地方
                # 需要更新三个Pose Encoder和Decoder
            reprojection_losses = torch.cat(reprojection_losses, 1)

            # 加上pose的loss
            # for idx, v, in enumerate(pose_loss):
            #     if times[idx] != 0:
            #         loss += self.opt.pose_weight * v / times[idx]


            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                #reprojection_losses.size() =12X2X192X640 

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda()) if torch.cuda.is_available() else   0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cpu())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                #doing_this
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                #doing_this
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                #doing this
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                #outputs["identity_selection/{}".format(scale)] = (
                outputs["identity_selection/{}".format(0)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)#defualt=1e-3 something with get_smooth_loss function
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        
        total_loss /= self.num_scales
        losses["loss"] = total_loss 
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so i#s only used to give an indication of validation performance


        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        #writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
