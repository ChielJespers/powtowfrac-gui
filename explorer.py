import gi
import os.path
import subprocess
import time
from tetration import tetr_execute

##############################################################################
## STATE
##############################################################################


##############################################################################
## SETUP
##############################################################################

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
builder = Gtk.Builder()
builder.add_from_file('explorer750.glade')

window = builder.get_object('MainWindow')

##############################################################################
## EVENTS
##############################################################################

def execute(button):    
    refresh()

def zoomin(button):
    epsilon = builder.get_object('epsilon')

    eps = float(epsilon.get_text())
    new_eps = eps * .7

    epsilon.set_text(str(new_eps))

    refresh()

def zoomout(button):
    epsilon = builder.get_object('epsilon')

    eps = float(epsilon.get_text())
    new_eps = eps * 1.4

    epsilon.set_text(str(new_eps))

    refresh()

def up(button):
    epsilon = builder.get_object('epsilon')
    imaginarypart = builder.get_object('imaginarypart')

    eps = float(epsilon.get_text())
    im = float(imaginarypart.get_text())
    new_im = im + .25*eps

    imaginarypart.set_text(str(new_im))

    refresh()

def down(button):
    epsilon = builder.get_object('epsilon')
    imaginarypart = builder.get_object('imaginarypart')

    eps = float(epsilon.get_text())
    im = float(imaginarypart.get_text())
    new_im = im - .25*eps

    imaginarypart.set_text(str(new_im))

    refresh()

def left(button):
    epsilon = builder.get_object('epsilon')
    realpart = builder.get_object('realpart')

    eps = float(epsilon.get_text())
    re = float(realpart.get_text())
    new_re = re - .25*eps

    realpart.set_text(str(new_re))

    refresh()

def right(button):
    epsilon = builder.get_object('epsilon')
    realpart = builder.get_object('realpart')

    eps = float(epsilon.get_text())
    re = float(realpart.get_text())
    new_re = re + .25*eps

    realpart.set_text(str(new_re))

    refresh()

def refresh():
    realpart = builder.get_object('realpart')
    imaginarypart = builder.get_object('imaginarypart')
    epsilon = builder.get_object('epsilon')
    maxiter = builder.get_object('maxiter')
    sharpness = builder.get_object('sharpness')

    re = realpart.get_text()
    im = imaginarypart.get_text()
    eps = epsilon.get_text()
    maxiter = maxiter.get_text()
    sharpness = sharpness.get_text()

    tetr_execute(re, im, eps, maxiter, sharpness, 'output.png')

    os.system('convert output.png -resize 750x750 output.png')
    #subprocess.Popen(['convert', 'output.png', '-resize', '750x750', 'output.png'])

    plot = builder.get_object('plot')
    plot.set_from_file('output.png')

def update_real(entry):
    pass

def update_imaginary(entry):
    pass

def update_epsilon(entry):
    pass

def update_sharpness(entry):
    pass

def exit_menu(menu):
    re = builder.get_object('realpart').get_text()
    im = builder.get_object('imaginarypart').get_text()
    eps = builder.get_object('epsilon').get_text()
    maxiter = builder.get_object('maxiter').get_text()
    sharpness = builder.get_object('sharpness').get_text()

    with open('session.txt', 'w') as f:
        f.write(re + '\n')
        f.write(im + '\n')
        f.write(eps + '\n')
        f.write(sharpness + '\n')
        f.write(maxiter + '\n')

    Gtk.main_quit()

def zoom_coords(box, event):
    print 'clicked on X=', event.x, ' Y= ', event.y

    epsilon = builder.get_object('epsilon')
    realpart = builder.get_object('realpart')
    imaginarypart = builder.get_object('imaginarypart')

    eps = float(epsilon.get_text())
    re = float(realpart.get_text())
    im = float(imaginarypart.get_text())

    new_eps = eps * .7
    new_re = re + ((event.x - 375) / 750) * eps
    new_im = im - ((event.y - 375) / 750) * eps

    realpart.set_text(str(new_re))
    imaginarypart.set_text(str(new_im))
    epsilon.set_text(str(new_eps))

    refresh()

handlers = {
    'on_exec-button_clicked': execute,
    'on_zoomin-button_clicked': zoomin,
    'on_zoomout-button_clicked': zoomout,
    'on_up-button_clicked': up,
    'on_left-button_clicked': left,
    'on_down-button_clicked': down,
    'on_right-button_clicked': right,
    'on_MainWindow_destroy': exit_menu,
    'on_realpart_changed': update_real,
    'on_imaginarypart_changed': update_imaginary,
    'on_epsilon_changed': update_epsilon,
    'on_sharpness_changed': update_sharpness,
    'on_plot-container-box_button_press_event': zoom_coords,
}
builder.connect_signals(handlers)

##############################################################################
## RUN APPLICATION
##############################################################################

if os.path.isfile('session.txt'):
    with open('session.txt', 'r') as f:
        content = [line.rstrip() for line in f]

        realpart = builder.get_object('realpart')
        imaginarypart = builder.get_object('imaginarypart')
        epsilon = builder.get_object('epsilon')
        sharpness = builder.get_object('sharpness')
        maxiter = builder.get_object('maxiter')

        realpart.set_text(content[0])
        imaginarypart.set_text(content[1])
        epsilon.set_text(content[2])
        sharpness.set_text(content[3])
        maxiter.set_text(content[4])

window.show_all()

Gtk.main()
