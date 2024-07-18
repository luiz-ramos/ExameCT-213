import numpy as np

def replace_color(data, original, new_value):
    r1, g1, b1 = original
    r2, g2, b2 = new_value

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]


def preprocess(img, greyscale=False):
    img = img.copy()
    # Remove numbers and enlarge speed bar
    for i in range(88, 93 + 1):
        img[i, 0:12, :] = img[i, 12, :]

    # Unify grass color
    replace_color(img, original=(102, 229, 102), new_value=(102, 204, 102))

    if greyscale:
        img = img.mean(axis=2)
        img = np.expand_dims(img, 2)

    # Make car black
    car_color = 68.0
    car_area = img[67:77, 42:53]
    car_area[car_area == car_color] = 0

    # Scale from 0 to 1
    img = img / img.max()

    # Unify track color
    img[(img > 0.411) & (img < 0.412)] = 0.4
    img[(img > 0.419) & (img < 0.420)] = 0.4

    # Change color of kerbs
    game_screen = img[0:83, :]
    game_screen[game_screen == 1] = 0.80
    return img


def decode_model_output(model_out):
    return np.array([model_out[0], model_out[1].clip(0, 1), -model_out[1].clip(-1, 0)])


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


