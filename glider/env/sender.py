import sys
import socket
import select
import time
from project_root import DIR
import keras_preprocessing as preproc
import matplotlib.pyplot as plt
from os import path
import numpy as np
import env.datagram_pb2
import Queue
from glidernorml.glider.helpers.helpers import (
    curr_ts_ms, apply_op, format_actions, class_vars, get_average,
    READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)
from gym import spaces

class Sender(object):
    def __init__(self, config, port):
        # pass the configure parameter
        self.config = config
        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        sys.stderr.write('[sender] Listening on port %s\n' %
                         self.sock.getsockname()[1])

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)
        self.curr_flags = ALL_FLAGS

        self.peer_addr = None

        self.dummy_payload = 'x' * 1400
        self.seq_num_que = Queue.Queue(1000000)  # sended packets queue
        self.loss_num = 0.  # total lost packet num
        self.send_num = 0  # total sent packet num
        self.packet_loss_rate = 0.  # loss packet rate

        self.cwnd = self.init_cwnd
        self.seq_num = 0
        self.next_ack = 0
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0
        self.rtt = float('inf')
        self.min_rtt = float('inf')
        self.delay_ewma = 0.
        self.send_rate_ewma = 0.
        self.delivery_rate_ewma = 0.
        self.last_delivered = 0
        self.last_duration = 0

        self.ts_first = None
        self.reward = 0.
        self.duration = 0
        self.tput = 0.

        self.step_start_ms = None


        self.rtt_buf = []
        self.tput_buf = []

        self.action_mapping = format_actions(self.action_list)

        self.send_cnt = 0
        self.recv_cnt = 0
        self.done = False
        self.step_end = False
        self.reset_flag = False

        '''for glider: set gym space'''
        highInf = np.array([#limit observation scope
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        highOnes = np.array([#limit observation scope
            1,
            1,
            1,
            1,
            1])
        self.observation_space = spaces.Box(-highInf, highInf, dtype=np.float32)
        #self.action_space = spaces.Discrete(5)
        self.action_space = spaces.Box(-highOnes,highOnes,dtype=np.float32)#gym like action space

    def cleanup(self):
        self.sock.close()


    def handshake(self):
        """Handshake with peer receiver. Must be called before run()."""

        while True:
            msg, addr = self.sock.recvfrom(1600)

            if msg == 'Hello from receiver' and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break

        self.sock.setblocking(0)  # non-blocking UDP socket

    def compute_performance(self):
        perc_delay = np.percentile(self.rtt_buf, 95)

        with open(path.join(DIR, 'results', 'performance'), 'w', 0) as perf:
            perf.write('%.2f %d\n' % (self.tput, perc_delay))

    def update_state(self, ack):
        """ Update the state variables listed in __init__() """
        # calculate packet loss rate
        if not self.seq_num_que.empty():
            seq_num = self.seq_num_que.get()  # get the ack_num of the last sent packet
            while not self.seq_num_que.empty() and seq_num < ack.seq_num:
                seq_num = self.seq_num_que.get()
                self.loss_num += 1

        if self.send_num == 0:
            self.packet_loss_rate = 0
        else:
            self.packet_loss_rate = self.loss_num / self.send_num

        self.next_ack = max(self.next_ack, ack.seq_num + 1)
        curr_time_ms = curr_ts_ms()

        # Update RTT
        self.rtt = float(curr_time_ms - ack.send_ts)
        self.min_rtt = min(self.min_rtt, self.rtt)

        # if self.reset_flag:
        #     self.ts_first = curr_time_ms
        #     self.reset_flag = False

        if self.reset_flag:
            self.ts_first = curr_time_ms
            self.reset_flag = False

        self.rtt_buf.append(self.rtt)


        delay = self.rtt - self.min_rtt
        if self.delay_ewma is None:
            self.delay_ewma = delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay

        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) /
                         max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma is None:
            self.delivery_rate_ewma = delivery_rate
        else:
            self.delivery_rate_ewma = (
                    0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)

        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, self.rtt)

        if self.send_rate_ewma is None:
            self.send_rate_ewma = send_rate
        else:
            self.send_rate_ewma = (
                    0.875 * self.send_rate_ewma + 0.125 * send_rate)

        self.duration = curr_time_ms - self.ts_first
        # throughput Mbps
        self.tput = 0.008 * (self.delivered - self.last_delivered) / max(1, self.duration)
        # self.tput = 0.008 * self.delivered / max(1, self.duration)
        self.tput_buf.append(self.tput)

    def update_reward(self):
        if len(self.tput_buf) < 2:
            self.reward = 0
        else:
            self.reward = self.alpha * (self.tput - self.tput_buf[-2]) \
                          - self.beta * (self.rtt - self.rtt_buf[-2]) \
                          - self.delta * self.packet_loss_rate

    def take_action(self, action_idx):
        op, val = self.action_mapping[action_idx]
        self.cwnd = int(apply_op(op, self.cwnd, val))
        self.cwnd = max(2.0, self.cwnd)
        self.cwnd = min(self.cwnd, 100000)

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        data = env.datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = curr_ts_ms()
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)

        self.seq_num += 1
        self.sent_bytes += len(serialized_data)

    def recv(self):
        # receive datagram
        serialized_ack, addr = self.sock.recvfrom(1600)

        if addr != self.peer_addr:
            return

        ack = env.datagram_pb2.Ack()
        ack.ParseFromString(serialized_ack)

        self.update_state(ack)
        self.update_reward()

        if self.step_start_ms is None:
            self.step_start_ms = curr_ts_ms()

        # At each step end, return info to agent
        # Elapsed a running time interval, then train the model
        if curr_ts_ms() - self.step_start_ms > self.step_len_ms:
            self.step_start_ms = curr_ts_ms()
            self.step_end = True

    def run(self):
        if self.window_is_open():
            if self.curr_flags != ALL_FLAGS:
                self.poller.modify(self.sock, ALL_FLAGS)
                self.curr_flags = ALL_FLAGS
        else:
            if self.curr_flags != READ_ERR_FLAGS:
                self.poller.modify(self.sock, READ_ERR_FLAGS)
                self.curr_flags = READ_ERR_FLAGS

        events = self.poller.poll(1000)  # TIMEOUT(ms)

        if not events:  # timed out
            self.send()
            self.send_cnt += 1


        for fd, flag in events:
            assert self.sock.fileno() == fd

            if flag & ERR_FLAGS:
                sys.exit('Error occurred to the channel')

            if flag & WRITE_FLAGS:
                if self.window_is_open():
                    self.send()
                    self.send_cnt += 1

            if flag & READ_FLAGS:
                self.recv()
                self.recv_cnt += 1

    def step(self, action):
        print("taking action...")
        print(action,'  ->  ',np.argmax(action))
        action = np.argmax(action)#for norml:action now is a 5dims arr
        self.take_action(action)
        while not self.step_end:
            self.run()
        self.step_end = False
        if len(self.tput_buf) > self.done_start:
            bad_tput_cnt = len([i for i in self.tput_buf[-self.last_entries_num:] if i <= self.bad_tput_threshold])
            bad_tput_fraction = float(bad_tput_cnt) / float(self.last_entries_num)
            if bad_tput_fraction > 0.9:
                self.done = True
        return self.state, self.reward, self.done, self.tput

    def reset(self):

        time.sleep(1)
        self.reset_flag = True
        self.cwnd = self.init_cwnd
        self.seq_num = 0
        self.next_ack = 0
        self.delivered_time = 0
        self.sent_bytes = 0
        self.rtt = float('inf')
        self.min_rtt = float('inf')
        self.delay_ewma = 0.
        self.send_rate_ewma = 0.
        self.delivery_rate_ewma = 0.

        self.reward = 0.
        self.tput = 0.

        self.step_start_ms = None

        self.tput_buf = []

        self.send_cnt = 0
        self.recv_cnt = 0
        self.done = False
        self.step_end = False

        # record at episode end
        self.last_delivered = self.delivered
        # self.last_duration = self.duration
        self.poller.register(self.sock, ALL_FLAGS)
        self.curr_flags = ALL_FLAGS
        return self.state

    @property
    def state(self):
        return np.array([self.delay_ewma,
                self.delivery_rate_ewma,
                self.send_rate_ewma,
                self.cwnd])

