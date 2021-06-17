# https://drive.google.com/file/d/1jzv4V1_8GzFuG6W3Ajd1FO8xeBi9RlEn/view?usp=sharing
fileId=1jzv4V1_8GzFuG6W3Ajd1FO8xeBi9RlEn
fileName=./datasets/MNIST.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
