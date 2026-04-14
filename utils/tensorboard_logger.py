from torch.utils.tensorboard import SummaryWriter
import os

# Global variable to hold the SummaryWriter instance
_tensorboard_writer = None

def setup_tensorboard(log_dir="runs"):
    """
    Sets up a global TensorBoard SummaryWriter.
    
    Parameters:
    - log_dir (str): Path to the TensorBoard log directory.
    """
    global _tensorboard_writer
    if _tensorboard_writer is None:
        os.makedirs(log_dir, exist_ok=True)
        _tensorboard_writer = SummaryWriter(log_dir=log_dir)

def get_tensorboard_logger():
    """
    Retrieves the global TensorBoard SummaryWriter.
    """
    global _tensorboard_writer
    if _tensorboard_writer is None:
        raise ValueError("TensorBoard logger has not been set up. Call `setup_tensorboard` first.")
    return _tensorboard_writer

def close_tensorboard():
    """
    Closes the global TensorBoard SummaryWriter.
    """
    global _tensorboard_writer
    if _tensorboard_writer:
        _tensorboard_writer.close()
        _tensorboard_writer = None
