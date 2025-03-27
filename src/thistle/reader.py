# class SatrecDict:
#     satrecs: dict[int, list[Satrec]]

#     def __init__(self) -> None:
#         self.satrecs = {}

#     def append(self, satrec: Satrec) -> None:
#         if satrec.satnum not in self.satrecs:
#             self.satrecs[satrec.satnum] = []
#         self.satrecs[satrec.satnum].append(satrec)

#     def extend(self, satrecs: list[Satrec]) -> None:
#         for satrec in satrecs:
#             self.append(satrec)

#     def get(self, satnum: Satnum) -> list[Satrec]:
#         satnum = from_alpha5(satnum)
#         if satnum not in self.satrecs:
#             return []
#         self.satrecs[satnum] = sorted(self.satrecs[satnum], key=_satrec_epoch)
#         return self.satrecs[satnum]


# class TLEReader:
#     satnums: Optional[list[int]]
#     _satrecs: SatrecDict

#     def __init__(self, satnums: Optional[list[Satnum]] = None) -> None:
#         self._satrecs = SatrecDict()
#         self.satnums = None
#         if satnums is not None:
#             self.satnums = [from_alpha5(satnum) for satnum in satnums]

#     def read(
#         self,
#         path: PathLike,
#     ) -> None:
#         tles = read_tle(path)
#         satrecs = _tles_to_satrecs(tles)

#         if self.satnums is not None:
#             satrecs = [sat for sat in satrecs if sat.satnum in self.satnums]

#         self._satrecs.extend(satrecs)

#     def select(self, satnum: Satnum) -> list[Satrec]:
#         return self._satrecs.get(satnum)
