python probe.py --dataset flowers --model_name llava7b --probe linear --split test --feature_type last
python probe.py --dataset flowers --model_name blip2 --probe linear --split test --feature_type last

python probe.py --dataset caltech --model_name llava7b --probe linear --split test --feature_type last
python probe.py --dataset caltech --model_name blip2 --probe linear --split test --feature_type last

python probe.py --dataset cars --model_name llava7b --probe linear --split test --feature_type last
python probe.py --dataset cars --model_name blip2 --probe linear --split test --feature_type last

python probe.py --dataset imagenet --model_name llava7b --probe linear --split valid --feature_type last --n_epochs 100
python probe.py --dataset imagenet --model_name blip2 --probe linear --split valid --feature_type last --n_epochs 100



python probe.py --dataset flowers --model_name llava7b --probe linear --split test --feature_type avg
python probe.py --dataset flowers --model_name blip2 --probe linear --split test --feature_type avg

python probe.py --dataset caltech --model_name llava7b --probe linear --split test --feature_type avg
python probe.py --dataset caltech --model_name blip2 --probe linear --split test --feature_type avg

python probe.py --dataset cars --model_name llava7b --probe linear --split test --feature_type avg
python probe.py --dataset cars --model_name blip2 --probe linear --split test --feature_type avg

python probe.py --dataset imagenet --model_name llava7b --probe linear --split valid --feature_type avg --n_epochs 100
python probe.py --dataset imagenet --model_name blip2 --probe linear --split valid --feature_type avg --n_epochs 100
