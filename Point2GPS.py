import math
PI = 3.1415926535898 
DistPerSec = 30.83

#point_x point_y，目标xy坐标， origin_longitude origin_latitude，激光雷达原点经纬度， direction_angle，y轴与正北方向逆时针夹角
def Point2GPS(point_x, point_y, origin_longitude, origin_latitude, direction_angle) :

    beta = math.atan2(point_x, point_y)
    if (0 > beta):
        beta += 2 * PI

    garma = direction_angle - beta
    if (0 > garma):
        garma += 2 * PI

    radius = math.sqrt(point_x * point_x + point_y * point_y)

    cal_latitude = radius * math.cos(garma) / DistPerSec
    cal_latitude = cal_latitude / 3600 + origin_latitude
	
    cal_longitude = radius * math.sin(garma) / (DistPerSec * math.cos((cal_latitude) * PI / 180))
    cal_longitude = cal_longitude / 3600 + origin_longitude

    return cal_longitude , cal_latitude


# before_longitude before_latitude，目标初始位置经纬度
# after_longitude after_latitude，目标终止位置经纬度
# time_difference 初始终止位置差时间 
def GPS2Speed(before_longitude, before_latitude, after_longitude, after_latitude, time_difference) :

    NS_point = (after_latitude - before_latitude) * 3600 * DistPerSec
    EW_point = (after_longitude - before_longitude)* 3600 * (DistPerSec * math.cos((after_latitude) * PI / 180))
	
    NS_speed = NS_point / time_difference
    EW_speed = EW_point / time_difference
	
    speed = math.sqrt(EW_speed * EW_speed + NS_speed * NS_speed)

    beta = math.atan2(EW_speed, NS_speed)
    if (0 > beta):
        beta += 2 * PI

    heading = beta * (180 / PI)

    return speed , heading
	

#test

# lon , lat = Point2GPS(30, 30, 121.23603816, 30.34010020, 0.0)
#
# print ("lon = ", lon)
# print ("lat = ", lat)

