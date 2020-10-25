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

    re = realpart.get_text()
    im = imaginarypart.get_text()
    eps = epsilon.get_text()
    maxiter = maxiter.get_text()

    tetr_execute(re, im, eps, maxiter)

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

def exit_menu(menu):
    re = builder.get_object('realpart').get_text()
    im = builder.get_object('imaginarypart').get_text()
    eps = builder.get_object('epsilon').get_text()
    maxiter = builder.get_object('maxiter').get_text()

    with open('session.txt', 'w') as f:
        f.write(re + '\n')
        f.write(im + '\n')
        f.write(eps + '\n')
        f.write(maxiter + '\n')

    Gtk.main_quit()

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
        maxiter = builder.get_object('maxiter')

        realpart.set_text(content[0])
        imaginarypart.set_text(content[1])
        epsilon.set_text(content[2])
        maxiter.set_text(content[3])

window.show_all()

Gtk.main()
