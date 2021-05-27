# StarField

![alt text](https://raw.githubusercontent.com/NaturalPatterns/StarField/master/starfield.gif)


## a nice-looking movie

``
python3 StarField.py --fname starfield --vext mp4 --fps 32 --dpi 100 --theta 0 --d_min 0.00001 --d_max 12 --bound_depth 20 --size 10 --mag 3 --T 10 --noise 0 --verbose --realistic  --N 100000
``

 * if you are looking for something more realistic, see for instance https://www.across-universe.com/

## a simple gif

``
python3 StarField.py --fname starfield --vext gif --fps 32  
``


## a folder with PNGs

``
python3 StarField.py --fname starfield --vext png --fps 32  
``


## a protocol


``
python3 StarField.py --fname 2020-03-05_starfield_A --vext png --fps 32 --theta 0.3 --seed 4201
python3 StarField.py --fname 2020-03-05_starfield_B --vext png --fps 32 --theta 0.15 --seed 4200  
python3 StarField.py --fname 2020-03-05_starfield_C --vext png --fps 32 --theta 0 --seed 420398348
``
