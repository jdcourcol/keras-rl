#!/usr/bin/env python
''' '''
import argparse
import race_agents


def main(args):
    ''' main '''
    for agent_name, agent_fct in race_agents.POSSIBLE_AGENTS.items():
        filename = agent_name + '_weight.h5f'
        env = race_agents.create_environment()
        agent = agent_fct(env)
        if args.fit:
            race_agents.fit_agent(env, agent, filename)
        if args.test:
            if args.log:
                env.log_file_name = agent_name + '_log.csv'
            agent.load_weights(filename)
            race_agents.test_agent(env, agent)


def get_parser():
    ''' return argparse parser '''
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument(
        '--fit',
        action='store_true', )
    parser.add_argument(
        '--test',
        action='store_true', )
    parser.add_argument(
        '--log',
        action='store_true')
    return parser


if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())
