"""
controller.py
=============
PD 제어기 모듈.

τ_i(t) = Kp_i * (θ_des_i(t) - θ_i(t)) + Kd_i * (θ̇_des_i(t) - θ̇_i(t))

토크 포화(saturation) 옵션 포함.
"""

from __future__ import annotations

import numpy as np


class PDController:
    """
    단일 모터에 대한 PD 제어기.

    Parameters
    ----------
    Kp : float  위치 게인 [Nm/rad]
    Kd : float  속도 게인 [Nm·s/rad]
    torque_limit : float | None  최대 토크 절댓값 [Nm]. None → 무한대.
    """

    def __init__(
        self,
        Kp: float,
        Kd: float,
        torque_limit: float | None = None,
    ) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self.torque_limit = torque_limit

    def compute(
        self,
        theta_des:  float,
        theta:      float,
        thetadot_des: float,
        thetadot:   float,
    ) -> float:
        """
        PD 토크 계산.

        τ = Kp*(θ_des - θ) + Kd*(θ̇_des - θ̇)

        Parameters
        ----------
        theta_des, theta : float [rad]  목표각, 현재각
        thetadot_des, thetadot : float [rad/s]  목표각속도, 현재각속도

        Returns
        -------
        torque : float [Nm]
        """
        tau = self.Kp * (theta_des - theta) + self.Kd * (thetadot_des - thetadot)
        if self.torque_limit is not None:
            tau = float(np.clip(tau, -self.torque_limit, self.torque_limit))
        return tau


def build_pd_controllers(pd_gains, torque_limit: float | None) -> tuple[PDController, PDController]:
    """
    config.PDGains 로부터 motor1, motor2 PDController 객체를 생성한다.

    Parameters
    ----------
    pd_gains    : config.PDGains
    torque_limit: float | None

    Returns
    -------
    (ctrl_motor1, ctrl_motor2)
    """
    ctrl1 = PDController(pd_gains.Kp_motor1, pd_gains.Kd_motor1, torque_limit)
    ctrl2 = PDController(pd_gains.Kp_motor2, pd_gains.Kd_motor2, torque_limit)
    return ctrl1, ctrl2
