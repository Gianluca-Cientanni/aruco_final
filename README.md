# aruco_final

aruco_test_template.py is the code that allows for an immediate ArUco pose estimation by providing
a premade calibration at 1m from the camera.

Things you need to have:

1) Camera with a spec of the Basler ac12440 75um (5MP) with a 8mm lens focused at 1m range.
2) Print-outs of ArUco markers, this link provides a generator to create them:
   " https://tn1ck.github.io/aruco-print/ ", or alternatively use the .pdf saved in the template
   folder.
3) The code is preset to use 5cm sized ArUco squares, ensure you change the code accordingly (it's
   clearly commented where you do this in the code).
4) Pose estimation is most accurate at the focus distance (1m).


The standoff distances will be printed to the console and also saved to a .csv file.

Happy days.
