import logging
import sys
from datetime import datetime
import os


class TrainingLogger:
    """
    Singleton logger class for training that handles both console output and file logging.
    Only outputs from rank 0 in distributed training.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.rank = 0
            cls._instance.logger = None
            cls._instance.log_file_path = None
        return cls._instance
    
    def setup(self, args, master_process, ddp_rank=0):
        """
        Set up comprehensive logging to capture all output.
        
        Args:
            args: parsed command line arguments
            master_process: whether this is the master process
            ddp_rank: DDP rank
            
        Returns:
            str: log file path if created, None otherwise
        """
        self.rank = ddp_rank
        self.total_steps = args.num_iterations if hasattr(args, 'num_iterations') else 0
        
        if not master_process or not args.output_dir:
            return None
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(args.output_dir, f"training_{timestamp}.log")
        
        # Set logging level
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        
        # Configure handlers
        handlers = [logging.FileHandler(self.log_file_path, mode='w')]
        if args.log_to_console:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Force reconfiguration if already configured
        )
        
        self.logger = logging.getLogger('training')
        self.logger.info(f"Logging initialized. Log file: {self.log_file_path}")
        self.logger.info(f"Arguments: {args}")
        
        return self.log_file_path
    
    def print0(self, *args, **kwargs):
        """
        Print function that only prints on rank 0 and logs to file if available.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for print function
        """
        if self.rank != 0:
            return
        
        # Convert args to string
        message = ' '.join(str(arg) for arg in args)
        
        # Print to console
        # print(message, **kwargs)
        
        # Log to file if logger is available
        if self.logger:
            self.logger.info(message)
    
    def log_metrics(self, step, metrics_dict):
        """
        Log training metrics in a structured format.
        
        Args:
            step: training step
            metrics_dict: dictionary of metrics to log
        """
        if self.rank != 0 or not self.logger:
            return
        
        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
        self.logger.info(f"Step {step}/{self.total_steps} | {metrics_str}")


# Create global instance
training_logger = TrainingLogger()

# Export the print0 function for easy access
def print0(*args, **kwargs):
    """Global print0 function that uses the training logger."""
    training_logger.print0(*args, **kwargs)

def setup_logging(args, master_process, ddp_rank=0):
    """
    Convenience function to setup logging using the TrainingLogger.
    
    Args:
        args: parsed command line arguments
        master_process: whether this is the master process
        ddp_rank: DDP rank
        
    Returns:
        tuple: (training_logger, log_file_path)
    """
    log_file_path = training_logger.setup(args, master_process, ddp_rank)
    return training_logger, log_file_path