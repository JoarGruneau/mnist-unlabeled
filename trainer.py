import logging

from ignite.contrib.handlers import CustomPeriodicEvent
from ignite.engine import Engine, Events

from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
from tqdm import tqdm


class ModelTrainer():
    def __init__(self,
                 models,
                 optimizers,
                 generators,
                 device,
                 loss_func,
                 metrics,
                 log_step=1):
        self.models = models
        self.optimizers = optimizers
        self.generators = generators
        self.device = device
        self.loss_func = loss_func
        self.metrics = metrics
        self.log_step = log_step
        logging.getLogger("ignite.engine.engine.Engine").setLevel(
            logging.WARNING)
        self.change_model_mode(mode='train')
        self.epoch = 0
        self.create_engines()

    def create_engines(self):
        if self.generators.get('train') is not None:
            self.trainer = self.create_trainer()

        if self.generators.get('test') is not None:
            self.tester = self.create_evaluator('test')

    def train(self, epochs):
        self.trainer.run(self.generators['train'], max_epochs=epochs)
        self.pbar.close()

    def evaluate(self, mode):
        self.tester.run(self.generators[mode])
        metrics = {**self.epochs, **self.tester.state.metrics}
        return metrics

    def log_epoch(self, engine):
        log_map = {
            **engine.state.metrics,
        }
        if engine.state.mode == 'train':
            for key, value in self.epochs.items():
                log_map[key] = value
        else:
            log_map['mode'] = engine.state.mode
        log(log_map)

    def create_trainer(self):
        def train_step(engine, batch):
            state = get_state(batch, self.device)
            self.loss_func(self.models, self.device, state)
            state['dec_loss'].backward()
            for optimizer in self.optimizers:
                optimizer.zero_grad()
                optimizer.step()
            state['dec_loss'] = state['dec_loss'].detach().cpu().item()
            return state

        def update_epoch(engine):
            self.epoch += 1

        def log_pbar(engine):
            self.pbar.update(100)

        def start_trainer(engine):
            self.pbar = tqdm(initial=0,
                             leave=False,
                             total=len(self.generators['train']) *
                             self.log_step)
            engine.state.mode = 'train'

        trainer = Engine(train_step)
        self.attach_metrics(trainer, 'train')

        trainer.add_event_handler(Events.STARTED, start_trainer)
        trainer.add_event_handler(Events.ITERATION_STARTED(every=100),
                                  log_pbar)
        trainer.add_event_handler(Events.EPOCH_STARTED, update_epoch)
        return trainer

    def create_evaluator(self, mode):
        def eval_step(engine, batch):
            state = get_state(batch, self.device)
            for i, model in enumerate(self.models):
                self.change_model_mode(mode='eval')
                self.loss_func[i](self.models, self.device, self.comission,
                                  state, self.allocations[mode])
                loss_key = model.get_name() + '_loss'
                state[loss_key] = state[loss_key].detach().cpu().item()
                self.change_model_mode(mode='train')
            return state

        def start_evaluator(engine):
            engine.state.mode = mode

        evaluator = Engine(eval_step)
        self.attach_metrics(evaluator, mode)
        evaluator.add_event_handler(Events.STARTED, start_evaluator)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch)
        return evaluator

    def change_model_mode(self, mode='train'):
        for i, model in enumerate(self.models):
            if mode == 'train':
                model.train()
            else:
                model.eval()

    def attach_metrics(self, engine, mode):
        for model in self.models:
            for metric in self.metrics[f'{model.get_name()}_{mode}']:
                metric.attach(
                    engine, f'{model.get_name()}_{metric.get_metric_name()}')


def log_msg(log_map, msg=None):
    log_string = f'{msg} ' if msg is not None else ''
    log_args = []
    logger = logging.getLogger()
    for key in log_map.keys():
        log_string += key
        log_args.append(log_map[key])
        if type(log_map[key]) == float:
            length = max(log_map[key] // 1 + 10, 10)
            log_string += '{:^10.5f}'
        elif type(log_map[key]) == str:
            length = max(10, len(log_map[key]) + 4)
            log_string += '{:^' + str(length) + '}'
        else:
            log_string += '{:^10}'
    logger.info(log_string.format(*log_args))


def get_state(batch, device):
    x, y = batch
    x, y = x.float().to(device), y.float().to(device)
    return {'data': x, 'labels': y}
