### Build Docker image
`docker build . -t phashion`

### Run docker image (locally after built)
`docker run -it --rm -v ~/Downloads/sample-city-park-400x300.jpg:/app/sample-city-park-400x300.jpg -e IMAGE_PATH=/app/sample-city-park-400x300.jpg phashion`