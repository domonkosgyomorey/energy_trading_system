class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_offers = (
            []
        )  # Temporary storage for offers before adding to the chain
        self.offers = {}  # {day: [list of offers]}
        self.reservations = {}  # {day: {household_id: reserved_offer}}
        self.used_offers = {}  # Track which offers have been used {day: [offer_ids]}

    def add_offer(self, day: int, offer: dict):
        self.pending_offers.append((day, offer))

    def finalize_day(self, day: int):
        for offer_day, offer in self.pending_offers:
            if offer_day not in self.offers:
                self.offers[offer_day] = []
            self.offers[offer_day].append(offer)
        self.pending_offers = []
        self.chain.append(f"Day {day} finalized")

    def get_offers(self, day: int, days_ahead: int):
        relevant_offers = []
        for d in range(day, day + days_ahead):
            if d in self.offers:
                for i, offer in enumerate(self.offers[d]):
                    if d not in self.used_offers or i not in self.used_offers[d]:
                        relevant_offers.append((i, offer))
        return relevant_offers

    def reserve_offer(self, day: int, household_id: int, offer_id: int):
        if day not in self.reservations:
            self.reservations[day] = {}
        self.reservations[day][household_id] = offer_id

    def mark_offer_as_used(self, day: int, offer_id: int):
        if day not in self.used_offers:
            self.used_offers[day] = []
        self.used_offers[day].append(offer_id)

    def get_reservations(self, day: int):
        return self.reservations.get(day, {})
