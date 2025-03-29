// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EnergyMarket {
    struct Offer {
        uint256 day;
        uint256 price;
        address provider;
        bool isAvailable;
    }

    mapping(uint256 => Offer[]) public offers;
    mapping(uint256 => mapping(address => uint256)) public reservations;

    // Add a new offer for a specific day
    function addOffer(uint256 day, uint256 price) public {
        Offer memory newOffer = Offer({
            day: day,
            price: price,
            provider: msg.sender,
            isAvailable: true
        });
        offers[day].push(newOffer);
    }

    // Reserve an offer for a specific day
    function reserveOffer(uint256 day, uint256 offerIndex) public {
        require(offers[day].length > offerIndex, "Offer not available");
        require(offers[day][offerIndex].isAvailable, "Offer already reserved");

        offers[day][offerIndex].isAvailable = false;
        reservations[day][msg.sender] = offerIndex;
    }

    // Get the available offers for a specific day
    function getOffers(uint256 day) public view returns (Offer[] memory) {
        return offers[day];
    }

    // Get reservation details for a specific day
    function getReservation(uint256 day) public view returns (uint256) {
        return reservations[day][msg.sender];
    }
}
