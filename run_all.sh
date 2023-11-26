#mkdir wkdir
#python main.py --config conf.json md --output wkdir/base_mnist.pkl
#python main.py --config conf.json tg --input wkdir/base_mnist.pkl --output wkdir/diff.pkl
#python main.py --config conf.json tc --input wkdir/base_mnist.pkl --output wkdir/base_cls.pkl
#python main.py --config conf.json cl --input wkdir/base_mnist.pkl --output wkdir/latent_mnist.pkl --model wkdir/diff.pkl
#python main.py --config conf.json tc --input wkdir/latent_mnist.pkl --output wkdir/latent_cls.pkl
#python main.py --config conf.json ti --input wkdir/base_mnist.pkl --output wkdir/base_img.pkl --idx 0 --mode test
#python main.py --config conf.json ca --input wkdir/base_img.pkl --output wkdir/base_adv_img.pkl --model wkdir/base_cls.pkl --label 0 --pmin 0.90
#python main.py --config conf.json ti --input wkdir/latent_mnist.pkl --output wkdir/latent_img.pkl --idx 0 --mode test
#python main.py --config conf.json ca --input wkdir/latent_img.pkl --output wkdir/latent_adv_img.pkl --model wkdir/latent_cls.pkl --label 0 --pmin 0.90
python main.py --config conf.json ri --input wkdir/latent_adv_img.pkl --output wkdir/latent_unsampled_adv_img.pkl --model wkdir/diff.pkl 
python main.py --config conf.json di --input wkdir/base_adv_img.pkl
python main.py --config conf.json di --input wkdir/latent_unsampled_adv_img.pkl
