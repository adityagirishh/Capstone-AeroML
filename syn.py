import csv
from io import StringIO
import math

# Placeholder for UAV maneuver data (replace with full phugoid_full_flap_push_entry.csv content)
data_str = """Time,Euler_Angle_Phi,Euler_Angle_Theta,Euler_Angle_Psi,Accel_x,Accel_y,Accel_z,Rot_Rate_x,Rot_Rate_y,Rot_Rate_z,Mag_x,Mag_y,Mag_z,Vel_x,Vel_y,Vel_z,V_tot_mps,Lat,Long,Easting,Northing,Alt,Battery_Voltage,Battery_Current,Motor_Rotation_Rate,Motor_PWM_Throttle,Aileron_defl,Elevator_defl,Rudder_del,Flap_defl,Airspeed,Pressure,Temp,A_x,A_y,A_z,Rot_Rate_x_filt,Rot_Rate_y_filt,Rot_Rate_z_filt,u_mps,v_mps,w_mps,alpha_deg,beta_deg
890.0000000078215,1.188965,-4.291384,-119.593231,0.049499,-0.728574,-9.18834,20.848673657661372,-8.859589090328406,-13.103143704185337,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,9.513194750879723,95654.0,10.0,0.009114925398882085,-0.028476657365758475,-8.50930699793157,21.260898443000464,-8.977757973671695,-13.287879801304253,,,,,
890.0025000078216,1.245373,-4.310798,-119.624557,0.15774,-0.273547,-8.373258,21.198177912691172,-8.1369174233299,-12.772031394379232,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.008999610668318808,-0.03344731909537836,-8.513088290623111,21.19585090078694,-9.019226710278662,-13.325652157821606,,,,,
890.0050000078216,1.303734,-4.333395,-119.655777,0.195372,-0.227464,-8.393268,21.977858880305195,-9.420973137997587,-12.6961144865244,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,9.676109490952026,95647.0,9.9375,0.009762628121942226,-0.0418103393183405,-8.523973087513097,21.12085900555639,-9.024489643731801,-13.358585654704848,,,,,
890.0075000078217,1.362894,-4.355527,-119.68811,-0.231245,0.453191,-9.000677,22.250644086566982,-9.260544955360956,-13.21063058655188,-0.077068,-0.478829,-1.061133,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.010598962382293785,-0.05210906682266918,-8.53213214789328,21.041579692565556,-9.042586366852396,-13.393866711726375,,,,,
890.0100000078218,1.421686,-4.372892,-119.720871,-0.215464,0.780564,-8.236434,22.09428390427578,-7.368638315838978,-13.338056400188973,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,6.706305536849367,2.5691510942459574,0.6955812637305833,45.09085763279031,,-1.0,-1.0,0.012970141324119028,-0.06001223389909007,-8.534790322776354,20.981380356106204,-9.068215047805685,-13.413013296137173,,,,,
890.0125000078218,1.478977,-4.395448,-119.754486,-0.007768,0.578189,-7.493621,21.462941709821123,-9.466981648946591,-13.624936368210976,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,9.83980395297577,95663.0,10.0,0.01252591876375648,-0.06535521313827156,-8.537753020487662,20.901591905428667,-9.08743097116627,-13.437832667353536,,,,,
890.0150000078219,1.53555,-4.421488,-119.788208,-0.128058,0.525059,-9.243442,21.166837121297515,-10.873421148654225,-13.616571184402067,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.01386625165537479,-0.06527624243362234,-8.536260070949274,20.84511993206117,-9.078909997269493,-13.47453846808554,,,,,
890.0175000078219,1.642112,-4.460707,-119.857147,0.37132,-0.239934,-7.949747,19.632341554378144,-8.525382808428597,-14.01999076795368,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.012514307816382615,-0.06252228126496646,-8.532920848302416,20.806298946542864,-9.097141111737537,-13.48869086960785,,,,,
890.020000007822,1.695291,-4.486364,-119.89151,0.213271,-0.85393,-8.873128,19.774320496011565,-10.765590491610604,-13.844722978423158,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,9.800738114694202,95676.25,10.0625,0.012300905321092476,-0.058007228179132245,-8.525899541209686,20.751562153603015,-9.122146781650475,-13.524352524452507,,,,,
890.022500007822,1.749045,-4.505329,-119.925598,-0.074968,-0.39778,-9.653789,20.00739972707078,-8.098242772158569,-13.804157566527898,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.009712892494828524,-0.06084461333488115,-8.52300733522837,20.653159643236712,-9.107188691658388,-13.566087845667898,,,,,
890.0250000078221,1.803377,-4.523156,-119.958511,0.280316,-0.494829,-8.204023,20.265345326438677,-7.641538113659789,-13.394664630347897,-0.079698,-0.493239,-1.063585,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,5.810050972083073,2.5637812940025313,0.6876700157616373,45.08253352534274,9.714878852022046,95659.25,9.9375,0.008324249169600858,-0.06204539616516086,-8.530630355298625,20.536408123098578,-9.12441804947685,-13.612288444426868,,,,,
890.0275000078221,1.858066,-4.546621,-119.991898,0.106072,0.112626,-7.489437,20.394432717681653,-9.914633574282304,-13.453793874805399,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.006518399794871682,-0.07171938374977375,-8.537881712574926,20.42415226248178,-9.108753311537432,-13.66357171086765,,,,,
890.0300000078222,1.915297,-4.568886,-120.024902,-0.042607,0.374449,-9.222445,21.417964522903354,-9.441542322842784,-13.30471025651236,0.0,0.0,0.0,,,,,,,,,,43.60431,0.0,357.08444371428567,0.00587666666666653,,,,,,-1.0,-1.0,0.008418522960577362,-0.0764085759891593,-8.532128095785893,20.277783126007634,-9.098677799097182,-13.717146625157568,,,,,
"""

# Parse UAV maneuver data
with StringIO(data_str) as f:
    reader = csv.reader(f)
    uav_headers = next(reader)
    maneuver_rows = [row for row in reader if row]

# Airplane log headers
airplane_headers = [
    "lcl date", "lcl time", "utcofst", "atvwpt", "latitude", "longitude", "altind", "baroa", "altmsl", "OAT",
    "ias", "gndspd", "vspd", "pitch", "roll", "latac", "normac", "hdg", "trk", "volt1", "volt2", "amp1", "amp2",
    "fqtyl", "fqtyr", "e1 fflow", "e1 oilt", "e1 oilp", "e1 rpm", "e1 cht1", "e1 cht2", "e1 cht3", "e1 cht4",
    "e1 egt1", "e1 egt2", "e1 egt3", "e1 egt4", "altgps", "tas", "hsis", "crs", "nav1", "nav2", "com1", "com2",
    "hcdi", "vcdi", "wndspd", "wnddr", "wptdst", "wptbrg", "magvar", "afcson", "rollm", "pitchm", "rollc",
    "pichc", "vspdg", "gpsfix", "hal", "val", "hplwas", "hplfd", "vplwas"
]

# Scale factors
s_time = 10.0
s_vel = 10.0  # ~20 m/s to 200 knots (1 m/s ≈ 1.94384 knots)
s_lin = 100.0
s_rot = 1.0 / s_time
base_alt = 9843.0  # ~3000 m
base_lat = 23.75205
base_long = 78.85717
ground_alt = 1722.0  # Bhopal airport elevation
base_uav_alt = 43.60431
pressure_scale = 0.88 ** (base_alt / 3280.84)  # Pressure adjustment for altitude
first_time = float(maneuver_rows[0][0])

# Scaled maneuver
maneuver_synthetic = []
for row in maneuver_rows:
    new_row = [''] * len(airplane_headers)
    time_diff = (float(row[0]) - first_time) * s_time
    seconds = int(time_diff)
    new_row[0] = "2021-01-19"
    new_row[1] = f"0 days 10:00:{seconds:02d}"
    new_row[4] = str(base_lat)  # Will update later
    new_row[5] = str(base_long)
    if row[21]:
        new_row[6] = str(ground_alt + (float(row[21]) - base_uav_alt) * s_lin * 0.3048)  # altind (ft)
        new_row[8] = str((float(row[21]) - base_uav_alt) * s_lin * 0.3048)  # altmsl (ft)
    else:
        new_row[6] = str(ground_alt)
        new_row[8] = "0.0"
    new_row[7] = "29.94"
    new_row[9] = "26.0"
    new_row[10] = str(float(row[30]) * s_vel * 1.94384 if row[30] else 200.0)  # ias (knots)
    new_row[11] = new_row[10]  # gndspd ≈ ias
    new_row[12] = str(float(row[15]) * s_vel * 196.85 if row[15] else 0.0)  # vspd (ft/min)
    new_row[13] = row[2]  # pitch = Euler_Angle_Theta
    new_row[14] = row[1]  # roll = Euler_Angle_Phi
    new_row[15] = str(float(row[5]) * s_vel / 9.81 if row[5] else 0.0)  # latac
    new_row[16] = str(float(row[6]) / 9.81 if row[6] else 0.0)  # normac
    new_row[17] = str((float(row[3]) + 360) % 360)  # hdg = Euler_Angle_Psi
    new_row[18] = new_row[17]  # trk ≈ hdg
    new_row[19] = "28.2"
    new_row[20] = "28.2"
    new_row[21] = "2.0"
    new_row[22] = "1.0"
    new_row[23] = "0.9"
    new_row[24] = "0.0"
    new_row[25] = "2.0"
    new_row[26] = "283.0"
    new_row[27] = "62.0"
    new_row[28] = str(float(row[24]) * 10 if row[24] else 1000.0)  # e1 rpm
    new_row[36] = str(base_lat)
    new_row[37] = str((float(row[21]) - base_uav_alt) * s_lin * 0.3048 if row[21] else 0.0)  # altgps = altmsl
    new_row[38] = str(base_long)
    new_row[39] = "0.0"  # grndtrk
    new_row[40] = "0.0"  # trueas
    new_row[41] = "108.1"
    new_row[42] = "122.6"
    new_row[43] = "122.6"
    new_row[48] = "0.1"
    new_row[49] = "0.0"
    new_row[59] = "3704.0"
    maneuver_synthetic.append(new_row)

# Takeoff (30 s, alt 1722->9843 ft, ias 0->200 knots)
dt = 1.0
takeoff_duration = 30.0
takeoff_steps = int(takeoff_duration / dt)
takeoff_rows = []
current_time = 0.0
current_alt = ground_alt
current_vel = 0.0
current_lat = base_lat
alt_gain = (base_alt - ground_alt) / takeoff_steps
vel_gain = 200.0 / takeoff_steps
for i in range(takeoff_steps):
    seconds = int(current_time)
    row = [''] * len(airplane_headers)
    row[0] = "2021-01-19"
    row[1] = f"0 days 09:59:{seconds:02d}"
    row[4] = str(current_lat)
    row[5] = str(base_long)
    row[6] = str(current_alt)
    row[7] = "29.94"
    row[8] = str(current_alt - ground_alt)
    row[9] = "26.0"
    row[10] = str(current_vel)
    row[11] = str(current_vel)
    row[12] = str(alt_gain * 60)  # vspd (ft/min)
    row[13] = "5.0"
    row[14] = "0.0"
    row[15] = "0.0"
    row[16] = "1.0"
    row[17] = "240.0"
    row[18] = "240.0"
    row[19] = "28.2"
    row[20] = "28.2"
    row[21] = "2.0"
    row[22] = "1.0"
    row[23] = "0.9"
    row[24] = "0.0"
    row[25] = "2.0"
    row[26] = "283.0"
    row[27] = "62.0"
    row[28] = "1000.0"
    row[37] = str(current_alt - ground_alt)
    row[38] = str(current_vel)
    row[39] = "352.0"
    row[40] = "108.1"
    row[41] = "108.1"
    row[42] = "122.6"
    row[43] = "122.6"
    row[48] = "0.1"
    row[49] = "0.0"
    row[59] = "3704.0"
    takeoff_rows.append(row)
    current_time += dt
    current_alt += alt_gain
    current_vel += vel_gain

# Cruise (5 min = 300 s, constant at base_alt)
cruise_duration = 300.0
cruise_steps = int(cruise_duration / dt)
cruise_rows = []
for i in range(cruise_steps):
    time_diff = current_time
    seconds = int(time_diff % 60)
    minutes = int(time_diff // 60)
    row = [''] * len(airplane_headers)
    row[0] = "2021-01-19"
    row[1] = f"0 days 10:{minutes:02d}:{seconds:02d}"
    row[4] = str(current_lat)
    row[5] = str(base_long)
    row[6] = str(base_alt)
    row[7] = "29.94"
    row[8] = str(base_alt - ground_alt)
    row[9] = "26.0"
    row[10] = "200.0"
    row[11] = "200.0"
    row[12] = "0.0"
    row[13] = "0.0"
    row[14] = "0.0"
    row[15] = "0.0"
    row[16] = "1.0"
    row[17] = "240.0"
    row[18] = "240.0"
    row[19] = "28.2"
    row[20] = "28.2"
    row[21] = "2.0"
    row[22] = "1.0"
    row[23] = "0.9"
    row[24] = "0.0"
    row[25] = "2.0"
    row[26] = "283.0"
    row[27] = "62.0"
    row[28] = "1000.0"
    row[37] = str(base_alt - ground_alt)
    row[38] = "200.0"
    row[39] = "352.0"
    row[40] = "108.1"
    row[41] = "108.1"
    row[42] = "122.6"
    row[43] = "122.6"
    row[48] = "0.1"
    row[49] = "0.0"
    row[59] = "3704.0"
    cruise_rows.append(row)
    current_time += dt
    current_lat += (200 * 1.852 / 3600) / 111  # Approximate lat change per second at 200 knots (1 knot = 1.852 km/h, 1 deg lat ~111 km)

# Offset maneuver time
maneuver_start_time = current_time
for row in maneuver_synthetic:
    time_diff = float(row[1].split(':')[-1]) + (maneuver_start_time - 30)  # Adjust for takeoff start
    seconds = int(time_diff % 60)
    minutes = int(time_diff // 60)
    row[1] = f"0 days 10:{minutes:02d}:{seconds:02d}"
    row[4] = str(current_lat)
    if row[10]:
        current_lat += (float(row[10]) * 1.852 / 3600) / 111
current_time = maneuver_start_time + (float(maneuver_rows[-1][0]) - first_time) * s_time

# Post-maneuver cruise (30 s at base_alt)
post_cruise_duration = 30.0
post_cruise_steps = int(post_cruise_duration / dt)
post_cruise_rows = []
for i in range(post_cruise_steps):
    time_diff = current_time
    seconds = int(time_diff % 60)
    minutes = int(time_diff // 60)
    row = [''] * len(airplane_headers)
    row[0] = "2021-01-19"
    row[1] = f"0 days 10:{minutes:02d}:{seconds:02d}"
    row[4] = str(current_lat)
    row[5] = str(base_long)
    row[6] = str(base_alt)
    row[7] = "29.94"
    row[8] = str(base_alt - ground_alt)
    row[9] = "26.0"
    row[10] = "200.0"
    row[11] = "200.0"
    row[12] = "0.0"
    row[13] = "0.0"
    row[14] = "0.0"
    row[15] = "0.0"
    row[16] = "1.0"
    row[17] = "240.0"
    row[18] = "240.0"
    row[19] = "28.2"
    row[20] = "28.2"
    row[21] = "2.0"
    row[22] = "1.0"
    row[23] = "0.9"
    row[24] = "0.0"
    row[25] = "2.0"
    row[26] = "283.0"
    row[27] = "62.0"
    row[28] = "1000.0"
    row[37] = str(base_alt - ground_alt)
    row[38] = "200.0"
    row[39] = "352.0"
    row[40] = "108.1"
    row[41] = "108.1"
    row[42] = "122.6"
    row[43] = "122.6"
    row[48] = "0.1"
    row[49] = "0.0"
    row[59] = "3704.0"
    post_cruise_rows.append(row)
    current_time += dt
    current_lat += (200 * 1.852 / 3600) / 111

# Descent (20 s, alt 9843->1722 ft, ias 200->20 knots)
descent_duration = 20.0
descent_steps = int(descent_duration / dt)
descent_rows = []
alt_gain = (ground_alt - base_alt) / descent_steps
vel_gain = (20.0 - 200.0) / descent_steps
for i in range(descent_steps):
    time_diff = current_time
    seconds = int(time_diff % 60)
    minutes = int(time_diff // 60)
    row = [''] * len(airplane_headers)
    row[0] = "2021-01-19"
    row[1] = f"0 days 10:{minutes:02d}:{seconds:02d}"
    row[4] = str(current_lat)
    row[5] = str(base_long)
    row[6] = str(current_alt)
    row[7] = "29.94"
    row[8] = str(current_alt - ground_alt)
    row[9] = "26.0"
    row[10] = str(current_vel)
    row[11] = str(current_vel)
    row[12] = str(alt_gain * 60)  # vspd (ft/min)
    row[13] = "-3.0"
    row[14] = "0.0"
    row[15] = "0.0"
    row[16] = "1.0"
    row[17] = "240.0"
    row[18] = "240.0"
    row[19] = "28.2"
    row[20] = "28.2"
    row[21] = "2.0"
    row[22] = "1.0"
    row[23] = "0.9"
    row[24] = "0.0"
    row[25] = "2.0"
    row[26] = "283.0"
    row[27] = "62.0"
    row[28] = "1000.0"
    row[37] = str(current_alt - ground_alt)
    row[38] = str(current_vel)
    row[39] = "352.0"
    row[40] = "108.1"
    row[41] = "108.1"
    row[42] = "122.6"
    row[43] = "122.6"
    row[48] = "0.1"
    row[49] = "0.0"
    row[59] = "3704.0"
    descent_rows.append(row)
    current_time += dt
    current_alt += alt_gain
    current_vel += vel_gain

# Landing (10 s, ground)
land_duration = 10.0
land_steps = int(land_duration / dt)
land_rows = []
for i in range(land_steps):
    time_diff = current_time
    seconds = int(time_diff % 60)
    minutes = int(time_diff // 60)
    row = [''] * len(airplane_headers)
    row[0] = "2021-01-19"
    row[1] = f"0 days 10:{minutes:02d}:{seconds:02d}"
    row[4] = str(current_lat)
    row[5] = str(base_long)
    row[6] = str(ground_alt)
    row[7] = "29.94"
    row[8] = "0.0"
    row[9] = "26.0"
    row[10] = "0.0"
    row[11] = "0.0"
    row[12] = "0.0"
    row[13] = "0.0"
    row[14] = "0.0"
    row[15] = "0.0"
    row[16] = "1.0"
    row[17] = "240.0"
    row[18] = "240.0"
    row[19] = "28.2"
    row[20] = "28.2"
    row[21] = "2.0"
    row[22] = "1.0"
    row[23] = "0.9"
    row[24] = "0.0"
    row[25] = "2.0"
    row[26] = "283.0"
    row[27] = "62.0"
    row[28] = "1000.0"
    row[37] = "0.0"
    row[38] = "0.0"
    row[39] = "352.0"
    row[40] = "108.1"
    row[41] = "108.1"
    row[42] = "122.6"
    row[43] = "122.6"
    row[48] = "0.1"
    row[49] = "0.0"
    row[59] = "3704.0"
    land_rows.append(row)
    current_time += dt

# Combine and save
all_rows = takeoff_rows + cruise_rows + maneuver_synthetic + post_cruise_rows + descent_rows + land_rows
with open('synthetic_phugoid_flight.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(airplane_headers)
    writer.writerows(all_rows)

print("Synthetic CSV saved to 'synthetic_phugoid_flight.csv'")

import pandas as pd

# Define parameters based on typical phugoid characteristics
# Total flight duration: 30 minutes (1800 seconds)
# Maneuver (phugoid oscillation) starts at 10 minutes (600 seconds)
# Maneuver duration: 5 minutes (300 seconds, allowing for multiple oscillation periods of ~60-90 seconds each)
total_duration = 1800  # seconds
maneuver_start_time = 600  # seconds
maneuver_duration = 300  # seconds
sample_rate = 1  # Hz, one sample per second

# Generate time strings and labels
time_strings = []
true_labels = []
current_time = 0  # start time in seconds

for t in range(total_duration + 1):
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    time_str = f"0 days 10:{minutes:02d}:{seconds:02d}"
    time_strings.append(time_str)
    
    if maneuver_start_time <= current_time < maneuver_start_time + maneuver_duration:
        true_labels.append(1)
    else:
        true_labels.append(0)
    
    current_time += sample_rate

# Create DataFrame
df = pd.DataFrame({
    'time': time_strings,
    'true_label': true_labels
})

# Save to CSV
df.to_csv('ground_truth_phugoid.csv', index=False)

print("Generated ground_truth_phugoid.csv with binary labels (1 for maneuver, 0 otherwise).")
print(f"Total rows: {len(df)}")
print(f"Maneuver rows: {sum(true_labels)}")