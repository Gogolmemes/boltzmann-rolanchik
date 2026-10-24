from boltzmann_lab.utils.resources import configure_cache


def main():
    configure_cache()

    from boltzmann_lab.app import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())