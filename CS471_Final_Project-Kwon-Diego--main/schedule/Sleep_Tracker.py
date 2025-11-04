import datetime
import time
import json
import pickle
import os

import schedule
import threading

class SleepTracker:
  def _init_(self):
    self.current_sleep_hours = 0

  def addSleep(self, hours):
    self.current_sleep_hours += hours

  def getSleep(self):
    return self.current_sleep_hours

class SleepManager:
  def _init_(self):
    self.sleepmanager = SleepTracker()

  def calculateSleep(self, start, end):
    current_sleep = end - start
    self.sleepmanager.addSleep(current_sleep)

  def correctSleepHours(self):
    current_hours = self.sleepmanager.getSleep()
    if current_hours >= 8:
      return True
    else:
      return False
