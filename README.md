# segment_spectrogram

# For Development

Create Image:
`docker build -t starling-server`

Run:
`docker run -itd --name starling --mount type=bind,source="$(pwd)/src/",dst=/app/src -p 5000:5000 starling-server`

To see logs,
`docker logs -f starling`

To see stats,
`docker stats`
