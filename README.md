# Autonomous Racecar

This project won our team 1st place in a 1:10-scale autonomous vehicle time trial that was the final project for UW EE545. The goal of the task was to autonomously navigate the course in as little time as possible, hitting all blue targets and avoiding all red targets without colliding with the environment.
#### Video:
https://youtu.be/nwsDgziJJ0Q

#### Team (Alphabetically):
- Bryan Tran
- James Harrang
- Rahul Ramanarayanan
- Russell DeGuzman

Professor: Patrick Lancaster
TA: Boling Yang

#### Implementation Details:
The car hardware is based on the MIT RACECAR (https://mit-racecar.github.io/) platform. Localization on a pre-generated (via SLAM) map is performed using a particle filter on LIDAR return data. Steering and acceleration is controlled via multiple dynamically-switching strategies. For more details, see our [full report](EE%20545%20Final%20Report.pdf).