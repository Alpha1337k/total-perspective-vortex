from joblib.numpy_pickle_utils import xrange
import pandas as pd
import argparse

import csp
import plot
import predict
import score
import train

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="total-perspective-vortex",
		description="Realtime EEG predictor"
	)

	subparser = parser.add_subparsers(dest="task")

	plot_parser = subparser.add_parser('plot')
	plot_parser.add_argument('--input', type=str, default='./data/S001/S001R01.edf')

	train_parser = subparser.add_parser('train')

	predict_parser = subparser.add_parser('predict')
	predict_parser.add_argument('-s', '--subject', type=int, choices=xrange(1,110), required=True)
	predict_parser.add_argument('-r', '--run', type=int, choices=xrange(3,15), required=True)

	csp_parser = subparser.add_parser('csp')
	score_parser = subparser.add_parser('score')

	args = parser.parse_args()


	match args.task:
		case "plot":
			plot.plot(args.input)
		case "train":
			train.train()
		case "csp":
			csp.run_csp()
		case "score":
			score.score()
		case "predict":
			predict.predict(args.subject, args.run)
		case _:
			parser.print_help()
			exit(1)