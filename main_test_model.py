from run_trained_model.run_trained_model import run_trained_model
from data_processing.region_type import RegionType

if __name__ == "__main__":

    # Inference for Novosibirsk region
    run_trained_model(RegionType.Nso)

    # Inference for Moscow
    run_trained_model(RegionType.Moscow)

    # Inference for Saint Petersburg
    run_trained_model(RegionType.Spb)