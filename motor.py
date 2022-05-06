#!/usr/bin/python3
"""NXT-Python tutorial: turn a motor."""
import nxt.locator
import nxt.motor
import nxt.sensor.generic

with nxt.locator.find() as b:
    # Get the motor connected to the port A.
    myMotor = b.get_motor(nxt.motor.Port.A)
    myMotor2 = b.get_motor(nxt.motor.Port.B)
    Motor = nxt.motor.SynchronizedMotors(myMotor, myMotor2, 1)
    myMotor.reset_position(0)
    sirCum = 17.6
    while True:
        # Wait for the next event.
        hvad = int(input(''' What do you want to do
         options =
         1: tacho
         2: drive a user-given distance Forward 
         3: drive a user-given distance Backward
         4: Forward and back at a user given distance'''))
        if hvad == 1:
            # extract data from the get tacho class
            motorValue = myMotor.get_tacho()
            st = str(motorValue)
            st = st.replace(")", "")
            x = st.split(", ")
            del x[0:2]
            x = int(x[0])
            # converts degrees to a distance
            dist = round(x*sirCum/360)
            print("You have driven %s cm" % dist)
        elif hvad == 2:
            dist = int(input("Distance[cm]? "))
            deg = int(360*dist/sirCum)
            Motor.turn(-120, deg)
            Motor.idle()
        elif hvad == 3:
            dist = int(input("Distance[cm]? "))
            deg = int(360*dist/sirCum)
            Motor.turn(120, deg)
            Motor.idle()
        elif hvad == 4:
            dist = int(input("Distance[cm]? "))
            deg = int(360 * dist / sirCum)
            Motor.turn(-85, deg)
            Motor.turn(85, deg)
            Motor.idle()
        else:
            print("Key not recognized")
