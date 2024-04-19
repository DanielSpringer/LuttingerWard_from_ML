import pytorch_lightning as L



class wrapped_AE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        module = __import__("models")
        self.model = getattr(module, config["MODEL_NAME"])(config)
        self.criterion_mse = nn.MSELoss()
        self.config = config
        self.val_pred = []
        self.val_loss = []


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, x)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, x)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = optimizer_str_to_obj(self)
        #TODO: expose scheduler parameters to config 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    def forward(self, batch: torch.Tensor):
        return self.model(batch)


    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])
