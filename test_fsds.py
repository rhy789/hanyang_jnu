#!/usr/bin/env python3
import rospy, csv, math, bisect, numpy as np
from collections import deque
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseStamped
from fs_msgs.msg import ControlCommand
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

class ADSControllerPurePursuit:
    def __init__(self):
        rospy.init_node('ads_controller_pp', anonymous=True)

        self.wheel_base     = rospy.get_param('~wheel_base',      1.6)
        self.target_speed   = rospy.get_param('~target_speed',    4.0)
        self.L_base         = rospy.get_param('~lookahead_base',  1.0)
        self.L_gain         = rospy.get_param('~lookahead_gain',  0.5)
        self.kp, self.ki, self.kd = (
            rospy.get_param('~kp', 0.8),
            rospy.get_param('~ki', 0.001),
            rospy.get_param('~kd', 0.2),
        )

        self.steer_smooth   = rospy.get_param('~steer_smooth', 0.8)
        self.prev_steer     = 0.0

        self.resample_dist     = rospy.get_param('~resample_dist',     0.5)
        self.smoothing_enabled = rospy.get_param('~smoothing_enabled', True)
        self.smoothing_window  = rospy.get_param('~smoothing_window',  21)

        self.csv_path = rospy.get_param(
            '~middle_line_path',
            './hand_driving_trainingmap.csv'
        )

        self.latlon_path   = []
        self.enu_path      = []
        self.ref_point     = None
        self.position      = None
        self.yaw           = None
        self.pos_hist      = deque(maxlen=2)
        self.current_speed = 0.0

        self.acc_error   = 0.0
        self.prev_error  = 0.0
        self.prev_time   = rospy.Time.now().to_sec()

        self.track_data  = []

        # Subscribers
        rospy.Subscriber('/fsds/gps', NavSatFix, self.gps_cb)
        rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, self.speed_cb)

        # Publishers
        self.ctrl_pub     = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        self.pose_pub     = rospy.Publisher('/current_pose', PoseStamped, queue_size=1)
        self.direction_pub= rospy.Publisher('/midline_direction_markers', MarkerArray, queue_size=1)

        self.load_and_preprocess_path()

        rospy.spin()


    def load_and_preprocess_path(self):
        with open(self.csv_path, newline='') as cf:
            rdr = csv.DictReader(cf)
            last = None
            for r in rdr:
                pt = (float(r['latitude']), float(r['longitude']))
                if pt != last:
                    self.latlon_path.append(pt)
                    last = pt

        self.latlon_path = self.resample(self.latlon_path, self.resample_dist)
        if self.smoothing_enabled and self.smoothing_window > 1:
            self.latlon_path = self.smooth(self.latlon_path, self.smoothing_window)


    def resample(self, pts, interval):
        R = 6378137.0
        lat0 = math.radians(pts[0][0])
        def dist(a, b):
            dlat = math.radians(b[0] - a[0])
            dlon = math.radians(b[1] - a[1])
            north = R * dlat
            east  = R * dlon * math.cos(lat0)
            return math.hypot(north, east)

        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + dist(pts[i-1], pts[i]))
        total = cum[-1]
        if total < interval:
            return pts[:]

        n = int(math.floor(total/interval))
        out = []
        for i in range(n+1):
            s = i*interval
            idx = bisect.bisect_right(cum, s) - 1
            if idx >= len(pts)-1:
                out.append(pts[-1])
                break
            s0, s1 = cum[idx], cum[idx+1]
            r = (s - s0)/(s1 - s0)
            a0, a1 = pts[idx], pts[idx+1]
            out.append((
                a0[0] + r*(a1[0]-a0[0]),
                a0[1] + r*(a1[1]-a0[1])
            ))
        return out


    def smooth(self, path, window):
        half = window // 2
        sm = []
        N = len(path)
        for i in range(N):
            lo = max(0, i-half)
            hi = min(N, i+half+1)
            seg = path[lo:hi]
            avg0 = sum(p[0] for p in seg)/len(seg)
            avg1 = sum(p[1] for p in seg)/len(seg)
            sm.append((avg0, avg1))
        return sm


    def gps_cb(self, msg):
        lat, lon = msg.latitude, msg.longitude

        if self.ref_point is None:
            self.ref_point = (lat, lon)
            self.enu_path = self.latlon_to_enu(self.latlon_path, self.ref_point)

        self.position = self.latlon_to_enu([(lat,lon)], self.ref_point)[0]

        ps = PoseStamped()
        ps.header = Header(stamp=rospy.Time.now(), frame_id='fsds/FSCar')
        ps.pose.position.x, ps.pose.position.y = self.position

        self.pos_hist.append(self.position)
        if len(self.pos_hist)==2:
            dx = self.pos_hist[1][0] - self.pos_hist[0][0]
            dy = self.pos_hist[1][1] - self.pos_hist[0][1]
            self.yaw = math.atan2(dy, dx)

        self.control()
        self.publish_direction_markers()


    def speed_cb(self, msg):
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)


    def latlon_to_enu(self, pts, ref):
        R = 6378137.0
        lat0 = math.radians(ref[0])
        out = []
        for la, lo in pts:
            dn = math.radians(la - ref[0]) * R
            de = math.radians(lo - ref[1]) * math.cos(lat0) * R
            out.append((dn, de))
        return out


    def find_target(self):
        Ld = self.L_base + self.L_gain * self.current_speed
        idx = int(np.argmin([
            math.hypot(self.position[0]-x, self.position[1]-y)
            for x,y in self.enu_path
        ]))
        acc = 0.0
        for i in range(idx, len(self.enu_path)-1):
            x0,y0 = self.enu_path[i]
            x1,y1 = self.enu_path[i+1]
            acc += math.hypot(x1-x0, y1-y0)
            if acc >= Ld:
                return x1, y1
        return self.enu_path[-1]


    def control(self):
        if self.position is None or self.yaw is None:
            return

        tx, ty = self.find_target()
        dx, dy = tx - self.position[0], ty - self.position[1]
        path_yaw = math.atan2(dy, dx)
        ang = (path_yaw - self.yaw + math.pi) % (2*math.pi) - math.pi

        Ld = math.hypot(dx, dy)
        th = math.atan2(2*self.wheel_base*math.sin(ang), Ld)
        raw_steer = max(min(th,1.0), -1.0)

        steer_cmd = (
            self.steer_smooth * self.prev_steer
            + (1-self.steer_smooth) * raw_steer
        )
        self.prev_steer = steer_cmd

        err = self.target_speed - self.current_speed
        now = rospy.Time.now().to_sec()
        dt  = max(now - self.prev_time, 1e-3)
        self.prev_time = now
        self.acc_error += err * dt
        deriv = (err - self.prev_error) / dt
        self.prev_error = err
        out = self.kp*err + self.ki*self.acc_error + self.kd*deriv
        thr_cmd = max(min(out,1.0), 0.0)

        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.steering    = steer_cmd
        cmd.throttle    = thr_cmd
        cmd.brake       = 0.0
        self.ctrl_pub.publish(cmd)
    
    def publish_direction_markers(self):
        if self.position is None or self.yaw is None:
            return

        marker_array = MarkerArray()
        now = rospy.Time.now()
        cn, ce = self.position
        cosy, siny = math.cos(self.yaw), math.sin(self.yaw)

        idx = int(np.argmin([
            math.hypot(cn - x, ce - y) for x,y in self.enu_path
        ]))

        dist_list = [0.0]
        for i in range(idx, len(self.enu_path)-1):
            x0,y0 = self.enu_path[i]
            x1,y1 = self.enu_path[i+1]
            dist_list.append(dist_list[-1] + math.hypot(x1-x0, y1-y0))

        for i in range(1,8):
            target_s = i * 1.0
            if target_s > dist_list[-1]:
                break
            j = bisect.bisect_right(dist_list, target_s) - 1
            s0, s1 = dist_list[j], dist_list[j+1]
            ratio = (target_s - s0) / (s1 - s0)
            wn0, we0 = self.enu_path[idx+j]
            wn1, we1 = self.enu_path[idx+j+1]
            wn = wn0 + ratio*(wn1-wn0)
            we = we0 + ratio*(we1-we0)

            dn, de = wn-cn, we-ce
            xf =  cosy*dn + siny*de
            yf = -siny*dn + cosy*de

            m = Marker()
            m.header.frame_id = 'fsds/FSCar'
            m.header.stamp    = now
            m.ns, m.id        = 'direction', i
            m.type, m.action = Marker.CYLINDER, Marker.ADD
            m.pose.position.x = xf
            m.pose.position.y = -yf
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = 0,1,0,0.8
            m.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(m)

        self.direction_pub.publish(marker_array)


if __name__ == '__main__':
    try:
        ADSControllerPurePursuit()
    except rospy.ROSInterruptException:
        pass
