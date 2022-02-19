import logging
import torch

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    logger.debug("No GPU found!, Using CPU instead")
    device = torch.device("cpu")





