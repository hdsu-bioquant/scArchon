import anndata
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats, sparse
import torch

class Trainer:
    def __init__(self, model, learning_rate=0.001, batch_size=32, n_epochs=25, patience=20, device=None):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience  # Early stopping patience
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_loader, valid_loader=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = self.model.loss_function
        
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                data_c, data_p = batch
                data_c = data_c.to(self.device)
                data_p = data_p.to(self.device)

                optimizer.zero_grad()
                # outputs = self.model(data_c, data_p)
                # loss = criterion(*outputs, data_c, data_p)
                x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self.model(data_c, data_p)
                        
                        
                loss = criterion(data_c, data_p, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c)
                        
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            if valid_loader is not None:
                self.model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        data_c, data_p = batch
                        data_c = data_c.to(self.device)
                        data_p = data_p.to(self.device)

                        x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self.model(data_c, data_p)
                        
                        
                        loss = criterion(data_c, data_p, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c)
                        valid_loss += loss.item()

                avg_valid_loss = valid_loss / len(valid_loader)
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

                # Check early stopping condition
                if avg_valid_loss < best_loss:
                    best_loss = avg_valid_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
    
    def evaluate(self, data_loader):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for batch_data in data_loader:
                data_c, data_p = batch_data
                data_c = data_c.to(self.device)
                data_p = data_p.to(self.device)

                x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self.model(data_c, data_p)
                        
                        
                loss += self.model.loss_function(data_c, data_p, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c)
        return loss / len(data_loader)
                
                
def prepare_dataloader(adata, indices, batch_size):
    X_c = adata[indices].X.A if sparse.issparse(adata.X) else adata[indices].X
    y = adata[indices].obs['label_column_name'].values  # Replace 'label_column_name' with your actual label column name
    tensor_dataset = TensorDataset(torch.tensor(X_c, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
                
                
def stratified_k_fold_cv(h5ad_file, model_class, n_splits=5, random_state=None, key='condition', **model_kwargs):
    # Load the dataset
    adata = anndata.read_h5ad(h5ad_file)
    
    # Extract the features and labels
    X = adata.X  # Features
    y = adata.obs[key].values  # Labels (replace 'label_column_name' with your actual label column name)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Loop over each fold
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        
        train_loader = prepare_dataloader(adata, train_index, model_kwargs.get('batch_size', 32))
        valid_loader = prepare_dataloader(adata, valid_index, model_kwargs.get('batch_size', 32))
        
        model = model_class(**model_kwargs)
        trainer = Trainer(model, patience=model_kwargs.get('patience', 5), **model_kwargs)
        trainer.train(train_loader, valid_loader)