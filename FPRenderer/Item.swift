//
//  Item.swift
//  FPRenderer
//
//  Created by Natalie Cuthbert on 2023/09/28.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
