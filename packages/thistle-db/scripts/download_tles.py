from getpass import getpass

from spacetrack import SpaceTrackClient


def main():
    username = input("Space-Track Username: ")
    password = getpass("Space-Track Password: ")
    with SpaceTrackClient(identity=username, password=password) as st:
        results = st.gp_history(
            norad_cat_id=25544, orderby="epoch desc", format="tle", iter_lines=True
        )

        with open("tests/data/25544.txt", "w") as fp:
            for line in results:
                fp.write(line + "\n")


if __name__ == "__main__":
    main()
