set -ex

docker build -t total-perspective-vortex .

docker run --rm -it -v $PWD:/app total-perspective-vortex aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ data