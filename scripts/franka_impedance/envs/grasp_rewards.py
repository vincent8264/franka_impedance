"""Bounded grasp-and-lift rewards, one scalar per environment."""

import torch


def grasp_reward_terms(distance, clearance, actions, grasped=None):
    reach = 0.02 * torch.exp(-distance / 0.08)
    lift = 0.05 * (clearance / 0.10).clamp(0.0, 1.0)
    if grasped is None:
        # Temporary smoke-test shaping. Height alone gives no evidence of a grasp.
        lift *= 0.1
        contact = 0.0
    else:
        lift *= grasped
        contact = 0.03 * grasped.float()
    action_cost = 0.005 * actions.square().mean(dim=-1)
    return reach, contact, lift, action_cost


def grasp_reward(distance, clearance, actions, completion, step_dt, grasped=None):
    reach, contact, lift, action_cost = grasp_reward_terms(distance, clearance, actions, grasped)
    return step_dt * (reach + contact + lift - action_cost) + 0.5 * completion.float()
