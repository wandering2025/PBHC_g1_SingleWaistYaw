import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.base_angular_vel_scale = config["input"]["scale"]["base_angular_vel"]
            self.projected_gravity_scale = config["input"]["scale"]["projected_gravity"]
            self.dof_pos_scale = config["input"]["scale"]["dof_pos"]
            self.dof_vel_scale = config["input"]["scale"]["dof_vel"]
            self.dof_angular_acceleration_scale = config["input"]["scale"]["dof_angular_acceleration"]


            self.loss_weight_accel = config["loss"]["weight"]["loss_weight_accel"]
            self.loss_weight_vel = config["loss"]["weight"]["loss_weight_vel"]
            self.loss_weight_consistency = config["loss"]["weight"]["loss_weight_consistency"]


            self.window_size = config["train"]["window_size"]
            self.block_dims = config["train"]["block_dims"]
            self.dropout_rate = config["train"]["dropout_rate"]
            self.dt = config["train"]["dt"]
            self.learning_rate = config["train"]["learning_rate"]
            self.weight_decay = config["train"]["weight_decay"]
            self.batch_size = config["train"]["batch_size"]
            self.num_epochs = config["train"]["num_epochs"]
            self.total_patience = config["train"]["total_patience"]
            self.scheduler_patience = config["train"]["scheduler_patience"]
