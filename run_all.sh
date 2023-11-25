python main.py --config conf.json md --output wkdir/base_minst.pkl
#python main.py --config conf.json tg --input wkdir/base_minst.pkl --output wkdir/diff.pkl
#python main.py --config conf.json tc --input wkdir/base_minst.pkl --output wkdir/base_cls.pkl
#python main.py --config conf.json cl --input wkdir/base_minst.pkl --output wkdir/latent_mnist.pkl --model diff.pkl
#python main.py --config conf.json tc --input wkdir/latent_minst.pkl --output wkdir/latent_cls.pkl
#python main.py --config conf.json ti --input wkdir/base_minst.pkl --output wkdir/base_img.pkl --idx 0 --mode test
#python main.py --config conf.json ca --input wkdir/base_img.pkl --output wkdir/base_adv_img.pkl --model base_cls.pkl --label 0 --pmin 0.99
#python main.py --config conf.json ti --input wkdir/latent_minst.pkl --output wkdir/latent_img.pkl --idx 0 --mode test
#python main.py --config conf.json ca --input wkdir/latent_img.pkl --output wkdir/latent_adv_img.pkl --model latent_cls.pkl --label 0 --pmin 0.99
#python main.py --config conf.json ca --input wkdir/latent_img.pkl --output wkdir/latent_adv_img.pkl --model base_cls.pkl --label 0 --pmin 0.99
#python main.py --config conf.json ui --input wkdir/latent_adv_img.pkl --output wkdir/latent_unsampled_adv_img.pkl --model base_cls.pkl --label 0 --pmin 0.99
#python main.py --config conf.json di --input wkdir/base_adv_img.pkl
#python main.py --config conf.json di --input wkdir/latent_unsampled_adv_img.pkl
