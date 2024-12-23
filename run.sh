set -ex

docker build -t total-perspective-vortex . 2> /dev/null

docker run --rm -it -v $PWD:/app total-perspective-vortex python3 $@