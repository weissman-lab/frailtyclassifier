
import os
from configargparse import ArgParser
from utils.misc import sheepish_mkdir, send_message_to_slack
from _09_AL_predict import BatchPredictor

def main():
    '''
    pulls `how_many` notes.  Assumes that the unlabled prediction in _09_AL_predict.py has been run for this batch
    '''
    p = ArgParser()
    p.add("-b", "--batchstring", help="the batch number", type=str)
    p.add("--how_many", default=0, type=int)

    options = p.parse_args()
    batchstring = options.batchstring
    assert batchstring is not None
    outdir = f"{os.getcwd()}/output/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"
    sheepish_mkdir(f"{ALdir}final_model/preds")

    predictor = BatchPredictor(digits="all", batchstring=batchstring)
    _ = predictor.pull_best_notes(options.how_many)
    predictor.write_notes()


if __name__ == "__main__":
    main()