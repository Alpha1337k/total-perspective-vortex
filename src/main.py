import pandas as pd
import argparse

import csp
import plot
import train

def load_df(input):
	df = pd.read_csv(args.input)

	labels = df['diagnosis'].copy()

	df.drop(labels=[ 'diagnosis' ], axis=1)

	return df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="total-perspective-vortex",
		description="Realtime EEG predictor"
	)

	subparser = parser.add_subparsers(dest="task")

	plot_parser = subparser.add_parser('plot')
	plot_parser.add_argument('--input', type=str, default='./data/S001/S001R01.edf')

	train_parser = subparser.add_parser('train')
	
	# train_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')
	# train_parser.add_argument('-i', '--iterations', default='200', type=int)
	# train_parser.add_argument('--learning-rate', default='0.5', type=float)
	# train_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='output/weights.json')
	# train_parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.py')


	split = subparser.add_parser('split')

	split.add_argument('-i', '--input', type=argparse.FileType('r'), default='data/raw.csv')
	split.add_argument('-v', '--validation-pct', type=int, default=20, choices=range(0, 100))
	split.add_argument('--train-path', type=str, default="./data/train.csv")
	split.add_argument('--validate-path', type=str, default="./data/validate.csv")

	test_parser = subparser.add_parser('train_test')
	test_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')

	predict_parser = subparser.add_parser('predict')
	predict_parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.py')
	predict_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/validate.csv')
	predict_parser.add_argument('-w', '--weights', type=argparse.FileType('r'), default='./output/weights.json')

	csp_parser = subparser.add_parser('csp')


	args = parser.parse_args()


	match args.task:
		case "plot":
			plot.plot(args.input)
		case "train":
			train.train()
		case "csp":
			csp.run_csp()
		# case "split":
		# 	split_dataset(args.input, args.validation_pct, args.train_path, args.validate_path)
		# case "train":
		# 	train.run(args.input, args.output, args.config, args.iterations, args.learning_rate)
		# case "predict":
		# 	predict.predict(args.weights, args.input, args.config)
		case _:
			parser.print_help()
			exit(1)